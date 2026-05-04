# ============================================================
# 视频抽帧服务
# 职责：从视频中按时间间隔抽取帧图片，创建图片集
# ============================================================
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from app.core.config import DEFAULT_EXTRACT_SECONDS
from app.core.state import AppState
from app.core.utils import now_iso, sanitize_filename
from app.services.task_manager import JobContext


class ExtractService:
    # 执行视频抽帧任务：按时间间隔提取帧图片，并创建图片集
    @staticmethod
    def run_extract_job(
        state: AppState,
        job_ctx: JobContext,
        video_id: str,
        sample_every_seconds: float = DEFAULT_EXTRACT_SECONDS,
        imageset_name: str | None = None,
        operator: str = "anonymous",
        creator_id: str = "",
    ) -> dict:
        video = state.get_video(video_id)
        if not video:
            raise ValueError("video_id 不存在")

        sample_every_seconds = max(0.01, float(sample_every_seconds))
        cap = cv2.VideoCapture(video.path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video.path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 25.0
        interval = max(1, int(round(fps * sample_every_seconds)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        frame_idx = 0
        saved = 0
        image_files: list[Path] = []

        with TemporaryDirectory(prefix=f"{job_ctx.job_id}_extract_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            try:
                while True:
                    job_ctx.raise_if_cancelled()
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_idx % interval == 0:
                        out = tmp_path / f"frame_{saved:06d}.jpg"
                        cv2.imwrite(str(out), frame)
                        image_files.append(out)
                        saved += 1

                    frame_idx += 1
                    if frame_idx % max(1, total_frames // 50) == 0:
                        job_ctx.set_progress(frame_idx, total_frames, f"抽帧中: {saved} 张")
            finally:
                cap.release()

            if not image_files:
                raise RuntimeError("抽帧结果为空，请检查视频内容")

            imageset = state.create_imageset(
                name=imageset_name or f"extract_{sanitize_filename(Path(video.filename).stem, 'video')}",
                source=f"video:{video_id}",
                image_files=image_files,
                creator_id=creator_id,
            )

            return {
                "video_id": video_id,
                "operator": (operator or "anonymous").strip() or "anonymous",
                "operator_id": creator_id,
                "operator_username": (operator or "anonymous").strip() or "anonymous",
                "imageset_id": imageset.id,
                "imageset_name": imageset.name,
                "sample_every_seconds": sample_every_seconds,
                "fps": fps,
                "saved_images": saved,
                "created_at": now_iso(),
            }
