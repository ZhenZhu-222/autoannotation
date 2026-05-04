# ============================================================
# 应用状态管理 (DAO 层 + Repository)
# 职责：内存中维护视频/模型/图片集的增删改查，并持久化到 JSON 文件
# 思路：
#   1. 启动时从磁盘加载全量数据到内存
#   2. 每次修改后立即写回磁盘
#   3. 用 RLock 保证线程安全
# ============================================================
from __future__ import annotations

import json
import shutil
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List

from app.core import config
from app.core.config import (
    IMAGESETS_DIR,
    MODELS_DIR,
    UPLOAD_VIDEOS_DIR,
    from_data_relative_path,
    to_data_relative_path,
)
from app.core.models import ImageRecord, ImageSetRecord, ModelRecord, UserRecord, VideoRecord
from app.core.passwords import hash_password
from app.core.utils import ensure_unique_path, new_id, now_iso, sanitize_filename

REVIEW_STATUSES = {"todo", "in_progress", "reviewed", "rejected", "accepted"}


class AppState:
    # ---------- 初始化与数据加载 ----------
    def __init__(self) -> None:
        self._lock = RLock()
        self.videos: Dict[str, VideoRecord] = {}
        self.models: Dict[str, ModelRecord] = {}
        self.imagesets: Dict[str, ImageSetRecord] = {}
        self.users: Dict[str, UserRecord] = {}
        self.image_index: Dict[str, tuple[str, ImageRecord]] = {}
        self._load_all()

    @property
    def videos_index_file(self) -> Path:
        return UPLOAD_VIDEOS_DIR / "index.json"

    # 启动时加载全量数据（视频、模型、图片集）
    def _load_all(self) -> None:
        self._load_users()
        self._load_videos()
        self._load_models()
        self._load_imagesets()

    @property
    def users_index_file(self) -> Path:
        return config.USERS_INDEX_FILE

    def _admin_user_id(self) -> str:
        return "user_admin"

    def _load_users(self) -> None:
        # 升级兼容：老版本没有 users/index.json 时，自动创建初始 admin。
        # 如果文件已经存在，绝不覆盖，避免客户改过的密码被重置。
        index_file = self.users_index_file
        if index_file.exists():
            try:
                payload = json.loads(index_file.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            for item in payload.get("users", []):
                item.setdefault("enabled", True)
                item.setdefault("created_at", "")
                item.setdefault("last_login_at", "")
                item.setdefault("deleted", False)
                record = UserRecord(**item)
                self.users[record.id] = record
        if not self.users:
            admin = UserRecord(
                id=self._admin_user_id(),
                username=config.ADMIN_USERNAME,
                password_hash=hash_password(config.ADMIN_PASSWORD),
                role="admin",
                enabled=True,
                created_at=now_iso(),
                deleted=False,
            )
            self.users[admin.id] = admin
            self._save_users()

    def _save_users(self) -> None:
        self.users_index_file.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted((user.to_dict() for user in self.users.values()), key=lambda x: x.get("created_at", ""))
        self.users_index_file.write_text(
            json.dumps({"users": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_user(self, user_id: str) -> UserRecord | None:
        with self._lock:
            return self.users.get(user_id)

    def get_user_by_username(self, username: str, *, include_deleted: bool = False) -> UserRecord | None:
        username = (username or "").strip()
        with self._lock:
            for user in self.users.values():
                if user.username == username and (include_deleted or not user.deleted):
                    return user
            return None

    def list_users(self, *, include_deleted: bool = False) -> List[UserRecord]:
        with self._lock:
            rows = [u for u in self.users.values() if include_deleted or not u.deleted]
            return sorted(rows, key=lambda x: x.created_at, reverse=True)

    def _ensure_username_available(self, username: str, existing_user_id: str = "") -> None:
        # 用户删除采用软删除。deleted=true 的旧用户不参与唯一性检查，
        # 这样删除 zhangsan 后，可以重新创建一个新的 zhangsan。
        for user in self.users.values():
            if user.id == existing_user_id or user.deleted:
                continue
            if user.username == username:
                raise ValueError("username already exists")

    def create_user(self, username: str, password: str, role: str = "operator") -> UserRecord:
        username = (username or "").strip()
        role = (role or "operator").strip()
        if not username:
            raise ValueError("username required")
        if len(password or "") < 6:
            raise ValueError("password too short")
        if role not in {"admin", "operator"}:
            raise ValueError("invalid role")
        with self._lock:
            self._ensure_username_available(username)
            record = UserRecord(
                id=new_id("user"),
                username=username,
                password_hash=hash_password(password),
                role=role,
                enabled=True,
                created_at=now_iso(),
                deleted=False,
            )
            self.users[record.id] = record
            self._save_users()
            return record

    def update_user(
        self,
        user_id: str,
        *,
        username: str | None = None,
        role: str | None = None,
        enabled: bool | None = None,
    ) -> UserRecord:
        with self._lock:
            user = self.users.get(user_id)
            if user is None or user.deleted:
                raise KeyError("user not found")
            if username is not None:
                clean = username.strip()
                if not clean:
                    raise ValueError("username required")
                self._ensure_username_available(clean, existing_user_id=user_id)
                user.username = clean
            if role is not None:
                if role not in {"admin", "operator"}:
                    raise ValueError("invalid role")
                user.role = role
            if enabled is not None:
                user.enabled = bool(enabled)
            self._save_users()
            return user

    def soft_delete_user(self, user_id: str) -> UserRecord:
        # 软删除保留 UserRecord，历史任务里仍能追溯 operator_username。
        # 同时从图片集 assignee_ids 移除，避免不可登录用户继续占分派位。
        with self._lock:
            user = self.users.get(user_id)
            if user is None or user.deleted:
                raise KeyError("user not found")
            user.enabled = False
            user.deleted = True
            for imageset in self.imagesets.values():
                if user_id in imageset.assignee_ids:
                    imageset.assignee_ids = [x for x in imageset.assignee_ids if x != user_id]
                    imageset.assignee_states.pop(user_id, None)
                    self._save_imageset(imageset)
            self._save_users()
            return user

    def reset_password(self, user_id: str, password: str) -> UserRecord:
        if len(password or "") < 6:
            raise ValueError("password too short")
        with self._lock:
            user = self.users.get(user_id)
            if user is None or user.deleted:
                raise KeyError("user not found")
            user.password_hash = hash_password(password)
            self._save_users()
            return user

    def mark_user_login(self, user_id: str) -> None:
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                return
            user.last_login_at = now_iso()
            self._save_users()

    def set_imageset_assignees(
        self,
        imageset_id: str,
        user_ids: List[str],
        *,
        actor_id: str = "",
        actor_username: str = "",
    ) -> ImageSetRecord:
        # assignee_ids 只存 operator。
        # admin 本来就全局可见可操作，不允许写进分派列表，保持数据干净。
        with self._lock:
            imageset = self.imagesets.get(imageset_id)
            if imageset is None:
                raise KeyError("imageset not found")
            clean: list[str] = []
            for user_id in user_ids:
                if user_id in clean:
                    continue
                user = self.users.get(user_id)
                if user is None or user.deleted or not user.enabled or user.role != "operator":
                    raise ValueError("assignees must be enabled operators")
                clean.append(user_id)
            existing_states = dict(getattr(imageset, "assignee_states", {}) or {})
            next_states: dict[str, dict] = {}
            now = now_iso()
            for removed_user_id in [x for x in imageset.assignee_ids if x not in clean]:
                previous = dict(existing_states.get(removed_user_id) or {})
                if not previous:
                    continue
                removed_user = self.users.get(removed_user_id)
                # 用完整分派集合覆盖时，被移除的人也要进历史流水，
                # 否则管理员后来查不到“谁被取消了分派”。
                archived = dict(previous)
                archived["user_id"] = removed_user_id
                archived["username"] = removed_user.username if removed_user else ""
                archived["status"] = "canceled"
                archived["reviewed_at"] = archived.get("reviewed_at") or now
                archived["reviewer_id"] = actor_id
                archived["reviewer_username"] = actor_username
                imageset.assignee_history.append(archived)
            for user_id in clean:
                previous = dict(existing_states.get(user_id) or {})
                next_states[user_id] = {
                    "status": previous.get("status") if previous.get("status") in {"assigned", "submitted", "rejected"} else "assigned",
                    "assigned_at": previous.get("assigned_at") or now,
                    "assigned_by_id": previous.get("assigned_by_id") or actor_id,
                    "assigned_by_username": previous.get("assigned_by_username") or actor_username,
                    "submitted_at": previous.get("submitted_at") or "",
                    "reviewed_at": previous.get("reviewed_at") or "",
                    "submit_note": previous.get("submit_note") or "",
                    "review_note": previous.get("review_note") or "",
                    "reviewer_id": previous.get("reviewer_id") or "",
                    "reviewer_username": previous.get("reviewer_username") or "",
                }
            imageset.assignee_ids = clean
            imageset.assignee_states = next_states
            self._save_imageset(imageset)
            return imageset

    # 单独移除一个分派人员，转入历史流水并记录操作人
    def remove_imageset_assignee(
        self,
        imageset_id: str,
        user_id: str,
        *,
        actor_id: str = "",
        actor_username: str = "",
    ) -> ImageSetRecord:
        with self._lock:
            imageset = self.imagesets.get(imageset_id)
            if imageset is None:
                raise KeyError("imageset not found")
            state = imageset.assignee_states.pop(user_id, None)
            if state:
                user = self.users.get(user_id)
                archived = dict(state)
                archived["user_id"] = user_id
                archived["username"] = user.username if user else ""
                archived["status"] = "canceled"
                archived["reviewed_at"] = archived.get("reviewed_at") or now_iso()
                archived["reviewer_id"] = actor_id
                archived["reviewer_username"] = actor_username
                imageset.assignee_history.append(archived)
            imageset.assignee_ids = [x for x in imageset.assignee_ids if x != user_id]
            self._save_imageset(imageset)
            return imageset

    def submit_imageset_assignment(self, imageset_id: str, user_id: str, submit_note: str = "") -> ImageSetRecord:
        # operator 处理完成后提交验收；提交后仍保留访问权，等待 admin 验收。
        with self._lock:
            imageset = self.imagesets.get(imageset_id)
            if imageset is None:
                raise KeyError("imageset not found")
            if user_id not in imageset.assignee_ids:
                raise ValueError("assignment not active")
            user = self.users.get(user_id)
            if user is None or user.deleted or not user.enabled or user.role != "operator":
                raise ValueError("assignee must be an enabled operator")
            record = dict(imageset.assignee_states.get(user_id) or {})
            now = now_iso()
            record["status"] = "submitted"
            record["assigned_at"] = record.get("assigned_at") or now
            record["submitted_at"] = now
            record["submit_note"] = (submit_note or "").strip()
            record["reviewed_at"] = ""
            record["review_note"] = ""
            record["reviewer_id"] = ""
            record["reviewer_username"] = ""
            imageset.assignee_states[user_id] = record
            self._save_imageset(imageset)
            return imageset

    def review_imageset_assignment(
        self,
        imageset_id: str,
        user_id: str,
        *,
        approved: bool,
        review_note: str = "",
        reviewer_id: str = "",
    ) -> ImageSetRecord:
        # admin 验收通过后从 assignee_ids 移除，operator 的“我的任务”里会自动消失。
        # 驳回则保留分派关系，让 operator 继续返工。
        with self._lock:
            imageset = self.imagesets.get(imageset_id)
            if imageset is None:
                raise KeyError("imageset not found")
            if user_id not in imageset.assignee_ids:
                raise ValueError("assignment not active")
            user = self.users.get(user_id)
            if user is None or user.deleted:
                raise ValueError("assignee not found")
            reviewer = self.users.get(reviewer_id)
            record = dict(imageset.assignee_states.get(user_id) or {})
            now = now_iso()
            record["assigned_at"] = record.get("assigned_at") or now
            record["reviewed_at"] = now
            record["review_note"] = (review_note or "").strip()
            record["reviewer_id"] = reviewer_id
            record["reviewer_username"] = reviewer.username if reviewer else ""
            if approved:
                record["status"] = "accepted"
                archived = dict(record)
                archived["user_id"] = user_id
                archived["username"] = user.username
                imageset.assignee_history.append(archived)
                imageset.assignee_states.pop(user_id, None)
                imageset.assignee_ids = [x for x in imageset.assignee_ids if x != user_id]
            else:
                record["status"] = "rejected"
                imageset.assignee_states[user_id] = record
            self._save_imageset(imageset)
            return imageset

    # 从磁盘读取视频索引到内存
    def _load_videos(self) -> None:
        if not self.videos_index_file.exists():
            return
        payload = json.loads(self.videos_index_file.read_text(encoding="utf-8"))
        for item in payload.get("videos", []):
            if item.get("path"):
                resolved = from_data_relative_path(item["path"])
                if resolved.is_absolute() and not resolved.exists():
                    fallback = UPLOAD_VIDEOS_DIR / Path(item["path"]).name
                    resolved = fallback
                item["path"] = str(resolved)
            record = VideoRecord(**item)
            self.videos[record.id] = record

    # 将内存中的视频记录写入磁盘索引文件
    def _save_videos(self) -> None:
        rows = []
        for video in self.videos.values():
            row = video.to_dict()
            row["path"] = to_data_relative_path(row["path"])
            rows.append(row)
        data = {"videos": rows}
        self.videos_index_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 扫描 MODELS_DIR，加载每个模型的 metadata.json 到内存
    def _load_models(self) -> None:
        for model_dir in MODELS_DIR.glob("model_*"):
            meta = model_dir / "metadata.json"
            if not meta.exists():
                continue
            payload = json.loads(meta.read_text(encoding="utf-8"))
            payload.setdefault("task", "detect")
            payload.setdefault("creator_id", self._admin_user_id())
            if payload.get("model_path"):
                resolved = from_data_relative_path(payload["model_path"])
                if resolved.is_absolute() and not resolved.exists():
                    resolved = MODELS_DIR / str(payload.get("id") or model_dir.name) / Path(payload["model_path"]).name
                payload["model_path"] = str(resolved)
            record = ModelRecord(**payload)
            self.models[record.id] = record

    # 重新扫描模型目录，拾取外部新增的模型文件
    def rescan_models(self) -> None:
        """Re-scan models directory to pick up externally added models."""
        with self._lock:
            self._load_models()

    # 将模型记录写入磁盘 metadata.json
    def _save_model(self, record: ModelRecord) -> None:
        model_dir = MODELS_DIR / record.id
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict()
        payload["model_path"] = to_data_relative_path(payload["model_path"])
        (model_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 扫描 IMAGESETS_DIR，加载每个图片集的 metadata.json 及图片记录到内存
    def _load_imagesets(self) -> None:
        for imageset_dir in IMAGESETS_DIR.glob("imageset_*"):
            meta = imageset_dir / "metadata.json"
            if not meta.exists():
                continue
            data = json.loads(meta.read_text(encoding="utf-8"))
            canonical_dir = imageset_dir.resolve()
            if data.get("dir_path"):
                resolved = from_data_relative_path(data["dir_path"])
                if resolved.is_absolute():
                    try:
                        inside_current_imagesets = resolved.resolve().is_relative_to(IMAGESETS_DIR.resolve())
                    except Exception:
                        inside_current_imagesets = False
                    if not resolved.exists() or not inside_current_imagesets:
                        resolved = canonical_dir
                else:
                    resolved = canonical_dir
                data["dir_path"] = str(resolved)
            else:
                data["dir_path"] = str(canonical_dir)
            images = [ImageRecord(**item) for item in data.get("images", [])]
            record = ImageSetRecord(
                id=data["id"],
                name=data["name"],
                source=data["source"],
                dir_path=data["dir_path"],
                created_at=data["created_at"],
                images=images,
                label_task=str(data.get("label_task") or "detect"),
                creator_id=str(data.get("creator_id") or self._admin_user_id()),
                assignee_ids=list(data.get("assignee_ids") or []),
                assignee_states=dict(data.get("assignee_states") or {}),
                assignee_history=list(data.get("assignee_history") or []),
            )
            self.imagesets[record.id] = record
            for image in images:
                self.image_index[image.id] = (record.id, image)

    # 将图片集记录写入磁盘 metadata.json
    def _save_imageset(self, record: ImageSetRecord) -> None:
        imageset_dir = Path(record.dir_path)
        imageset_dir.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict()
        payload["dir_path"] = to_data_relative_path(payload["dir_path"])
        (imageset_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- 视频 CRUD ----------
    # 注册新视频记录并持久化
    def register_video(self, filename: str, path: str) -> VideoRecord:
        with self._lock:
            record = VideoRecord(id=new_id("video"), filename=filename, path=path, created_at=now_iso())
            self.videos[record.id] = record
            self._save_videos()
            return record

    # 按 ID 查询视频记录
    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._lock:
            return self.videos.get(video_id)

    # ---------- 模型 CRUD ----------
    # 注册新模型记录并持久化到磁盘
    def register_model(
        self,
        name: str,
        model_type: str,
        model_path: str,
        classes: List[str],
        class_source: str,
        task: str = "detect",
        creator_id: str = "",
    ) -> ModelRecord:
        with self._lock:
            record = ModelRecord(
                id=new_id("model"),
                name=name,
                model_type=model_type,
                model_path=model_path,
                classes=classes,
                class_source=class_source,
                created_at=now_iso(),
                task=task or "detect",
                creator_id=creator_id or self._admin_user_id(),
            )
            self.models[record.id] = record
            self._save_model(record)
            return record

    # 按 ID 查询模型记录
    def get_model(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            return self.models.get(model_id)

    # 返回所有模型记录（按创建时间倒序）
    def list_models(self) -> List[ModelRecord]:
        with self._lock:
            return sorted(self.models.values(), key=lambda x: x.created_at, reverse=True)

    # 修改模型名称并持久化
    def rename_model(self, model_id: str, new_name: str) -> dict:
        with self._lock:
            model = self.models.get(model_id)
            if model is None:
                raise KeyError("model not found")
            model.name = new_name.strip()
            self._save_model(model)
            return {"model_id": model_id, "name": model.name}

    # 删除模型记录及磁盘目录
    def delete_model(self, model_id: str) -> dict:
        with self._lock:
            model = self.models.pop(model_id, None)
            if model is None:
                raise KeyError("model not found")
            model_dir = MODELS_DIR / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            return {"model_id": model_id, "deleted": True}

    # ---------- 图片集 CRUD ----------
    # 创建图片集：拷贝图片到 IMAGESETS_DIR，生成记录并持久化
    def create_imageset(
        self,
        name: str,
        source: str,
        image_files: Iterable[Path],
        creator_id: str = "",
    ) -> ImageSetRecord:
        with self._lock:
            imageset_id = new_id("imageset")
            imageset_dir = IMAGESETS_DIR / imageset_id
            images_dir = imageset_dir / "images"
            labels_dir = imageset_dir / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            records: List[ImageRecord] = []
            for path in image_files:
                src_path = Path(path)
                if not src_path.exists():
                    continue
                safe_name = sanitize_filename(src_path.name, "image")
                dst_path = ensure_unique_path(images_dir / safe_name)
                if src_path.resolve() != dst_path.resolve():
                    shutil.copy2(src_path, dst_path)
                rel_path = dst_path.relative_to(imageset_dir).as_posix()
                image = ImageRecord(
                    id=new_id("img"),
                    imageset_id=imageset_id,
                    filename=dst_path.name,
                    rel_path=rel_path,
                    created_at=now_iso(),
                )
                records.append(image)

            record = ImageSetRecord(
                id=imageset_id,
                name=name,
                source=source,
                dir_path=str(imageset_dir),
                created_at=now_iso(),
                images=records,
                label_task="detect",
                creator_id=creator_id or self._admin_user_id(),
                assignee_ids=[],
                assignee_states={},
                assignee_history=[],
            )
            self.imagesets[record.id] = record
            for image in record.images:
                self.image_index[image.id] = (record.id, image)
            self._save_imageset(record)
            return record

    # 按 ID 查询图片集记录
    def get_imageset(self, imageset_id: str) -> ImageSetRecord | None:
        with self._lock:
            return self.imagesets.get(imageset_id)

    # 返回所有图片集记录（按创建时间倒序）
    def list_imagesets(self) -> List[ImageSetRecord]:
        with self._lock:
            return sorted(self.imagesets.values(), key=lambda x: x.created_at, reverse=True)

    # 修改图片集名称并持久化
    def rename_imageset(self, imageset_id: str, new_name: str) -> dict:
        with self._lock:
            imageset = self.imagesets.get(imageset_id)
            if imageset is None:
                raise KeyError("imageset not found")
            imageset.name = new_name.strip()
            self._save_imageset(imageset)
            return {"imageset_id": imageset_id, "name": imageset.name}

    # 删除图片集记录及磁盘目录
    def delete_imageset(self, imageset_id: str) -> dict:
        with self._lock:
            imageset = self.imagesets.pop(imageset_id, None)
            if imageset is None:
                raise KeyError("imageset not found")
            for image in imageset.images:
                self.image_index.pop(image.id, None)
            imageset_dir = Path(imageset.dir_path)
            if imageset_dir.exists():
                shutil.rmtree(imageset_dir, ignore_errors=True)
            return {"imageset_id": imageset_id, "deleted": True}

    # 计算图片的完整磁盘路径
    def resolve_image_path(self, imageset_id: str, image: ImageRecord) -> Path:
        return Path(self.imagesets[imageset_id].dir_path) / image.rel_path

    # ---------- 图片 CRUD ----------
    # 删除单张图片及对应标签文件，并更新图片集记录
    def delete_image(self, image_id: str) -> dict:
        with self._lock:
            if image_id not in self.image_index:
                raise KeyError("image not found")

            imageset_id, image = self.image_index.pop(image_id)
            imageset = self.imagesets[imageset_id]
            image_path = Path(imageset.dir_path) / image.rel_path
            stem = image_path.stem
            deleted = {"image": False, "label": False}

            if image_path.exists():
                image_path.unlink()
                deleted["image"] = True

            label_candidates = [
                image_path.with_suffix(".txt"),
                Path(imageset.dir_path) / "labels" / f"{stem}.txt",
            ]
            for candidate in label_candidates:
                if candidate.exists():
                    candidate.unlink()
                    deleted["label"] = True

            imageset.images = [item for item in imageset.images if item.id != image_id]
            self._save_imageset(imageset)
            return {
                "image_id": image_id,
                "imageset_id": imageset_id,
                "deleted": deleted,
            }

    # 批量删除图片
    def delete_images(self, image_ids: List[str]) -> dict:
        results = []
        for image_id in image_ids:
            try:
                results.append({"ok": True, **self.delete_image(image_id)})
            except KeyError:
                results.append({"ok": False, "image_id": image_id, "error": "not found"})
        return {"results": results}

    # ---------- 图片审核 ----------
    # 更新图片审核状态（todo/in_progress/reviewed/rejected/accepted）
    def update_image_review(
        self,
        image_id: str,
        review_status: str,
        reviewer: str = "",
        review_note: str = "",
    ) -> dict:
        status = str(review_status or "").strip().lower()
        if status not in REVIEW_STATUSES:
            raise ValueError(f"review_status 仅支持: {', '.join(sorted(REVIEW_STATUSES))}")

        with self._lock:
            if image_id not in self.image_index:
                raise KeyError("image not found")
            imageset_id, image = self.image_index[image_id]
            imageset = self.imagesets[imageset_id]
            image.review_status = status
            image.reviewer = (reviewer or "").strip()
            image.review_note = (review_note or "").strip()
            image.reviewed_at = now_iso()
            self._save_imageset(imageset)
            return {
                "image_id": image.id,
                "imageset_id": imageset_id,
                "review_status": image.review_status,
                "reviewer": image.reviewer,
                "review_note": image.review_note,
                "reviewed_at": image.reviewed_at,
            }

    # 统计图片集各审核状态的数量
    def imageset_review_summary(self, imageset_id: str) -> dict:
        with self._lock:
            imageset = self.imagesets.get(imageset_id)
            if not imageset:
                raise KeyError("imageset not found")
            counts = {k: 0 for k in sorted(REVIEW_STATUSES)}
            for image in imageset.images:
                status = image.review_status if image.review_status in REVIEW_STATUSES else "todo"
                counts[status] += 1
            return {
                "imageset_id": imageset_id,
                "total": len(imageset.images),
                "counts": counts,
            }
