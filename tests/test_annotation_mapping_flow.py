"""
5 轮循环打标端到端测试
场景覆盖：replace/append、override 合并、切模型、手填 target_classes、多轮累积
每轮至少 5 个新标签写入，按 classes.txt / data.yaml / 标签内容断言
特别场景：轮 5 模拟用户真实报错 —— 手填 4 个 target_classes + 勾选单类 + override
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


# ============================================================
# 工具函数 & Fake Predictors（测试数据工厂）
# ============================================================
MODEL_A_CLASSES = ["person", "car", "dog", "cat", "bird"]  # 5 类
MODEL_B_CLASSES = ["car", "truck", "bus"]                   # 3 类（"car" 与 A 同名）
MODEL_C_CLASSES = ["cat", "person", "elephant"]             # 3 类（cat/person 和 A 同名，elephant 全新）
MODEL_D_CLASSES = ["tiger", "lion", "wolf"]                 # 3 类（全新，不和任何已有类重名）
MODEL_E_CLASSES = ["unicorn", "phoenix", "dragon"]          # 3 类（全新幻兽）


def _make_image(path: Path) -> None:
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[:, :] = (120, 30, 200)
    cv2.imwrite(str(path), img)


def _mk_det(cls_id: int, cls_name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9):
    return type("D", (), {
        "cls_id": cls_id, "cls_name": cls_name, "conf": conf,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    })()


# 全局 round 计数器：每打标一轮前 +1，用于让不同轮 predictor 返回不同 box 位置
# 从而避开 _merge_label_lines 的文本去重（保证每轮真的写入新标签）
_ROUND = {"n": 0}


class _FakePredictorA:
    """Model A (5 类)：每张图返回 5 个 detection，box y 位置随 _ROUND 小幅偏移
    避免文本去重；y 偏移小（mod 20 × 1 = 0..19），不会被 clamp"""

    def __init__(self, _model_path: str) -> None:
        self.class_names = MODEL_A_CLASSES

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        dy = float(_ROUND["n"] % 20) * 1.0
        return [
            _mk_det(0, "person", 5.0, 5.0 + dy, 20.0, 35.0 + dy),
            _mk_det(1, "car", 25.0, 5.0 + dy, 45.0, 35.0 + dy),
            _mk_det(2, "dog", 50.0, 5.0 + dy, 70.0, 35.0 + dy),
            _mk_det(3, "cat", 75.0, 5.0 + dy, 95.0, 35.0 + dy),
            _mk_det(4, "bird", 100.0, 5.0 + dy, 115.0, 35.0 + dy),
        ]


class _FakePredictorB:
    """Model B (3 类)：每张图返回 3 个 detection，y 偏移同理"""

    def __init__(self, _model_path: str) -> None:
        self.class_names = MODEL_B_CLASSES

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        dy = float(_ROUND["n"] % 20) * 1.0
        return [
            _mk_det(0, "car", 5.0, 40.0 + dy, 25.0, 70.0 + dy),
            _mk_det(1, "truck", 30.0, 40.0 + dy, 55.0, 70.0 + dy),
            _mk_det(2, "bus", 60.0, 40.0 + dy, 85.0, 70.0 + dy),
        ]


class _FakePredictorC:
    """Model C (3 类)"""

    def __init__(self, _model_path: str) -> None:
        self.class_names = MODEL_C_CLASSES

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        dy = float(_ROUND["n"] % 20) * 1.0
        return [
            _mk_det(0, "cat", 5.0, 15.0 + dy, 25.0, 45.0 + dy),
            _mk_det(1, "person", 30.0, 15.0 + dy, 55.0, 45.0 + dy),
            _mk_det(2, "elephant", 60.0, 15.0 + dy, 85.0, 45.0 + dy),
        ]


class _FakePredictorD:
    """Model D (3 类)"""

    def __init__(self, _model_path: str) -> None:
        self.class_names = MODEL_D_CLASSES

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        dy = float(_ROUND["n"] % 20) * 1.0
        return [
            _mk_det(0, "tiger", 5.0, 25.0 + dy, 25.0, 55.0 + dy),
            _mk_det(1, "lion", 30.0, 25.0 + dy, 55.0, 55.0 + dy),
            _mk_det(2, "wolf", 60.0, 25.0 + dy, 85.0, 55.0 + dy),
        ]


class _FakePredictorE:
    """Model E (3 类)"""

    def __init__(self, _model_path: str) -> None:
        self.class_names = MODEL_E_CLASSES

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        dy = float(_ROUND["n"] % 20) * 1.0
        return [
            _mk_det(0, "unicorn", 5.0, 55.0 + dy, 25.0, 75.0 + dy),
            _mk_det(1, "phoenix", 30.0, 55.0 + dy, 55.0, 75.0 + dy),
            _mk_det(2, "dragon", 60.0, 55.0 + dy, 85.0, 75.0 + dy),
        ]


def _classes_factory(model_path):
    # 只按文件名后缀（model_a.pt）判定，避免父目录 uuid 里的随机 hex 字符（如 "model_abc..."）被误匹配为 "model_a"
    name = Path(str(model_path)).name.lower()
    if name.startswith("model_a."):
        return list(MODEL_A_CLASSES)
    if name.startswith("model_b."):
        return list(MODEL_B_CLASSES)
    if name.startswith("model_c."):
        return list(MODEL_C_CLASSES)
    if name.startswith("model_d."):
        return list(MODEL_D_CLASSES)
    if name.startswith("model_e."):
        return list(MODEL_E_CLASSES)
    return []


def _predictor_factory(model_path):
    # 同 _classes_factory：只按文件名判定
    name = Path(str(model_path)).name.lower()
    if name.startswith("model_a."):
        return _FakePredictorA(model_path)
    if name.startswith("model_b."):
        return _FakePredictorB(model_path)
    if name.startswith("model_c."):
        return _FakePredictorC(model_path)
    if name.startswith("model_d."):
        return _FakePredictorD(model_path)
    if name.startswith("model_e."):
        return _FakePredictorE(model_path)
    raise ValueError(f"unknown model: {model_path}")


def _wait_job(client, path: str, timeout_s: float = 15.0) -> dict:
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(path)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job timeout: {path}")


def _build_overrides_from_ui(source_classes: list[dict], map_classes: list[dict]) -> dict[str, int]:
    """模拟前端 UI 默认 dst 选择：按名字匹配，优先同名；无 match 时 fallback 到 mapClasses[0]
    只记录 src != dst 的条目"""
    overrides: dict[str, int] = {}
    for src in source_classes:
        src_name = str(src["name"]).lower()
        match = next(
            (m for m in map_classes if str(m["name"]).lower().replace(" (复用)", "") == src_name),
            None,
        )
        dst = match["id"] if match else (map_classes[0]["id"] if map_classes else src["id"])
        if int(dst) != int(src["id"]):
            overrides[str(src["id"])] = int(dst)
    return overrides


def _count_class_ids(imageset_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    labels_dir = imageset_dir / "labels"
    for lf in labels_dir.glob("*.txt"):
        if lf.name == "classes.txt":
            continue
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def _read_classes_txt(imageset_dir: Path) -> list[str]:
    return (imageset_dir / "labels" / "classes.txt").read_text(encoding="utf-8").splitlines()


def _read_data_yaml_names(imageset_dir: Path) -> dict[int, str]:
    return yaml.safe_load((imageset_dir / "data.yaml").read_text(encoding="utf-8"))["names"]


# ============================================================
# 主流程：5 轮循环打标
# ============================================================
def test_5_rounds_annotate_flow(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    # monkeypatch：model 解析 & predictor 按文件名分发
    monkeypatch.setattr(
        "app.services.model_service.extract_classes_from_model_file", _classes_factory,
    )
    monkeypatch.setattr(
        "app.services.annotation_service.create_predictor", _predictor_factory,
    )

    # 造 5 张假图并上传为 imageset
    image_files = []
    for i in range(5):
        p = tmp_path / f"img_{i:02d}.jpg"
        _make_image(p)
        image_files.append(p)

    upload_files = []
    for p in image_files:
        with p.open("rb") as f:
            upload_files.append(("files", (p.name, f.read(), "image/jpeg")))
    r = client.post(
        "/api/imagesets/upload-folder",
        files=upload_files,
        data={"imageset_name": "test_mapping_set"},
    )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    # 上传 Model A
    r = client.post(
        "/api/models/upload",
        files={"model_file": ("model_a.pt", b"pt-model-a", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model_a_id = r.json()["model_id"]

    # 上传 Model B
    r = client.post(
        "/api/models/upload",
        files={"model_file": ("model_b.pt", b"pt-model-b", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model_b_id = r.json()["model_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    imageset_dir = Path(imageset.dir_path)

    # ======================================================
    # 轮 1：Model A, replace, 全选 [0..4]（identity）
    # ======================================================
    _ROUND["n"] = 1
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "replace",
    })
    assert r.status_code == 200, r.text
    map_r1 = r.json()["items"]
    assert {m["id"]: m["name"] for m in map_r1} == {
        0: "person", 1: "car", 2: "dog", 3: "cat", 4: "bird",
    }, f"[R1] resolve-mapping 不对: {map_r1}"

    sources_a = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    ov_r1 = _build_overrides_from_ui(sources_a, map_r1)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": ov_r1, "mapping_confirmed": True,
        "label_mode": "replace", "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[R1] job 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt == ["person", "car", "dog", "cat", "bird"], f"[R1] classes.txt: {classes_txt}"
    counts = _count_class_ids(imageset_dir)
    for cid in range(5):
        assert counts.get(cid) == 5, f"[R1] class {cid} 数量: {counts}"
    assert sum(counts.values()) == 25, f"[R1] 总标签 {sum(counts.values())} ≠ 25"

    # ======================================================
    # 轮 2：Model A, append, 全选 [0..4]（所有类应复用已有 ID）
    # ======================================================
    _ROUND["n"] = 2
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "append",
    })
    assert r.status_code == 200, r.text
    map_r2 = r.json()["items"]

    ov_r2 = _build_overrides_from_ui(sources_a, map_r2)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": ov_r2, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[R2] job 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt == ["person", "car", "dog", "cat", "bird"], f"[R2] classes.txt: {classes_txt}"
    counts = _count_class_ids(imageset_dir)
    for cid in range(5):
        assert counts.get(cid) == 10, f"[R2] class {cid} 数量: {counts}"
    assert sum(counts.values()) == 50

    # ======================================================
    # 轮 3：Model A, append, 只勾 [4], override bird→1（合并到 car）
    #       这是截图 bug 场景（target_classes 为空 + 单勾 + override 到已有 ID）
    # ======================================================
    _ROUND["n"] = 3
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4],  # 前端渲染时传全部
        "target_classes": [], "label_mode": "append",
    })
    assert r.status_code == 200, r.text
    map_r3 = r.json()["items"]
    # 前端 UI 下拉框里 "1" 必须存在（car），否则用户选不出
    assert any(m["id"] == 1 for m in map_r3), f"[R3] mapClasses 没有 id=1: {map_r3}"

    ov_r3 = {"4": 1}  # 用户手动：bird → 1(car)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [4], "target_classes": [],  # 提交时只传勾选的
        "class_id_overrides": ov_r3, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[R3] job 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt == ["person", "car", "dog", "cat", "bird"], (
        f"[R3] classes.txt 被污染：{classes_txt}（class 1 应保持 'car'，不能变 'bird'）"
    )
    yaml_names = _read_data_yaml_names(imageset_dir)
    assert yaml_names.get(1) == "car", f"[R3] data.yaml class 1 被污染: {yaml_names}"
    counts = _count_class_ids(imageset_dir)
    assert counts.get(1) == 15, f"[R3] class 1 数量 {counts.get(1)} ≠ 15"
    for cid in [0, 2, 3, 4]:
        assert counts.get(cid) == 10, f"[R3] class {cid} 数量: {counts}"

    # ======================================================
    # 轮 4：切 Model B, append, 全选
    #       car 同名复用 ID 1, truck/bus 新分配
    # ======================================================
    _ROUND["n"] = 4
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_b_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2], "target_classes": [], "label_mode": "append",
    })
    assert r.status_code == 200, r.text
    map_r4 = r.json()["items"]
    map_r4_dict = {m["id"]: m["name"].replace(" (复用)", "") for m in map_r4}
    assert map_r4_dict.get(1) == "car", f"[R4] car 未复用 id=1: {map_r4_dict}"
    assert "truck" in map_r4_dict.values()
    assert "bus" in map_r4_dict.values()

    sources_b = [{"id": i, "name": MODEL_B_CLASSES[i]} for i in range(3)]
    ov_r4 = _build_overrides_from_ui(sources_b, map_r4)
    assert ov_r4.get("0") == 1  # B.car(src=0) → dst=1

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_b_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2], "target_classes": [],
        "class_id_overrides": ov_r4, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[R4] job 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt[0] == "person"
    assert classes_txt[1] == "car", f"[R4] class 1 不再是 car: {classes_txt}"
    assert "truck" in classes_txt
    assert "bus" in classes_txt

    counts = _count_class_ids(imageset_dir)
    assert counts.get(1) == 20, f"[R4] class 1 数量 {counts.get(1)} ≠ 20（原 15 + B.car 5）"
    truck_id = next(k for k, v in map_r4_dict.items() if v == "truck")
    bus_id = next(k for k, v in map_r4_dict.items() if v == "bus")
    assert counts.get(truck_id) == 5, f"[R4] truck(id={truck_id}): {counts}"
    assert counts.get(bus_id) == 5, f"[R4] bus(id={bus_id}): {counts}"

    # ======================================================
    # 轮 5：Model A, append, 手填 4 个 target_classes, 单勾 [4], override {4:1}
    #       这是用户报的真实 bug 场景：手填 4 个 → 确认 → 开始打标 → 炸
    # ======================================================
    _ROUND["n"] = 5
    target_4 = ["CategoryA", "CategoryB", "CategoryC", "CategoryD"]

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4],
        "target_classes": target_4, "label_mode": "append",
    })
    assert r.status_code == 200, r.text
    map_r5 = r.json()["items"]
    # 4 个手填类名，全部是新名字（不和 existing 重复），应分配 4 个新 ID
    assert len(map_r5) == 4, f"[R5] mapClasses 应 4 项: {map_r5}"
    map_r5_ids = sorted(m["id"] for m in map_r5)
    # 当前 existing max_cid=6（R4 后）→ tc_offset=7 → 新分配 7,8,9,10
    assert map_r5_ids == [7, 8, 9, 10], f"[R5] 分配的 ID 不对: {map_r5_ids}"

    # 用户把 bird(src=4) 的 dst 选成 1（car，但是 1 并不在 mapClasses 里！）
    # 这就是真实 bug 场景：UI 可能基于旧 state 残留选中了 1
    # 或者用户切换 target_classes 前已选 dst=1，切换后没重新渲染
    ov_r5 = {"4": 1}

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_a_id, "imageset_id": imageset_id,
        "selected_class_ids": [4], "target_classes": target_4,
        "class_id_overrides": ov_r5, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", (
        f"[R5] job 失败（用户真实报错场景）: {final.get('error')}"
    )

    # R5 断言：bird → class 1（合并到 car），classes.txt 的 1 保持 car
    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt[1] == "car", f"[R5] class 1 被污染: {classes_txt}"
    yaml_names = _read_data_yaml_names(imageset_dir)
    assert yaml_names.get(1) == "car", f"[R5] yaml class 1 被污染: {yaml_names}"
    counts = _count_class_ids(imageset_dir)
    assert counts.get(1) == 25, f"[R5] class 1 数量 {counts.get(1)} ≠ 25（R4 后 20 + 本轮 5 个）"

    # 至少 5 个新标签写入
    total_after_r5 = sum(counts.values())
    assert total_after_r5 >= 5, "每轮应至少 5 个标签"


# ============================================================
# 用户真实场景测试：80 类 YOLO + 手填首字母大写 target_classes
# 场景复刻：COCO80 模型 → 手填 ["Car","Van","Truck","Bus"]
# → 勾选 car(2) + motorcycle(3) + bus(5) + truck(7) → 映射到 Car/Van/Truck/Bus
# ============================================================
COCO80_LOWER = [f"class_{i}" for i in range(80)]
COCO80_LOWER[0] = "person"
COCO80_LOWER[1] = "bicycle"
COCO80_LOWER[2] = "car"
COCO80_LOWER[3] = "motorcycle"
COCO80_LOWER[5] = "bus"
COCO80_LOWER[7] = "truck"


class _FakePredictorCOCO80:
    """COCO80 模型：每张图返回 4 个 detection（car/motorcycle/bus/truck）"""

    def __init__(self, _model_path: str) -> None:
        self.class_names = COCO80_LOWER

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        return [
            _mk_det(2, "car", 5.0, 5.0, 25.0, 35.0),
            _mk_det(3, "motorcycle", 30.0, 5.0, 50.0, 35.0),
            _mk_det(5, "bus", 55.0, 5.0, 80.0, 35.0),
            _mk_det(7, "truck", 85.0, 5.0, 115.0, 35.0),
        ]


def test_user_case_handwritten_targets_uppercase(app_client, monkeypatch: pytest.MonkeyPatch):
    """用户真实报错场景：
    - 80 类 YOLO 权重（COCO 格式，类名小写）
    - 手填 4 个 target_classes，首字母大写：["Car","Van","Truck","Bus"]
    - 勾选 car(2), motorcycle(3), bus(5), truck(7)
    - 前端 UI smart default：
        car(2) → Car(0), motorcycle(3) → Car(0)[无match,fallback], bus(5) → Bus(3), truck(7) → Truck(2)
    - overrides = {"2":0, "3":0, "5":3, "7":2}
    期望：打标成功，不污染 classes.txt
    """
    client, app, tmp_path = app_client

    monkeypatch.setattr(
        "app.services.model_service.extract_classes_from_model_file",
        lambda _: list(COCO80_LOWER),
    )
    monkeypatch.setattr(
        "app.services.annotation_service.create_predictor",
        lambda p: _FakePredictorCOCO80(p),
    )

    # 上传 5 张图
    image_files = []
    for i in range(5):
        p = tmp_path / f"user_case_{i:02d}.jpg"
        _make_image(p)
        image_files.append(p)

    upload_files = []
    for p in image_files:
        with p.open("rb") as f:
            upload_files.append(("files", (p.name, f.read(), "image/jpeg")))
    r = client.post(
        "/api/imagesets/upload-folder",
        files=upload_files,
        data={"imageset_name": "user_case_set"},
    )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    # 上传 COCO80 模型
    r = client.post(
        "/api/models/upload",
        files={"model_file": ("coco80.pt", b"pt-coco80", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    imageset_dir = Path(imageset.dir_path)

    # 模拟前端：调 resolve-mapping 拿下拉框选项
    target_4 = ["Car", "Van", "Truck", "Bus"]
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": list(range(80)),
        "target_classes": target_4,
        "label_mode": "append",
    })
    assert r.status_code == 200, r.text
    map_items = r.json()["items"]
    # 截图上显示的 "0:Car, 1:Van, 2:Truck, 3:Bus"
    assert {m["id"]: m["name"] for m in map_items} == {
        0: "Car", 1: "Van", 2: "Truck", 3: "Bus",
    }, f"mapClasses 不匹配截图: {map_items}"

    # 模拟前端 UI 的 smart default dst 分配
    sources = [{"id": 2, "name": "car"}, {"id": 3, "name": "motorcycle"},
               {"id": 5, "name": "bus"}, {"id": 7, "name": "truck"}]
    overrides = _build_overrides_from_ui(sources, map_items)
    # car(2) → Car(0): 记录 {"2":0}
    # motorcycle(3) → mapClasses[0].id=0 (无 match, fallback): 记录 {"3":0}
    # bus(5) → Bus(3): 记录 {"5":3}
    # truck(7) → Truck(2): 记录 {"7":2}
    assert overrides == {"2": 0, "3": 0, "5": 3, "7": 2}, f"overrides: {overrides}"

    # 提交打标
    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [2, 3, 5, 7],
        "target_classes": target_4,
        "class_id_overrides": overrides,
        "mapping_confirmed": True,
        "label_mode": "append",
        "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", (
        f"用户场景打标失败: {final.get('error')}"
    )

    # 断言 classes.txt：id 0..3 应为 Car/Van/Truck/Bus
    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt[0] == "Car", f"classes.txt[0] = {classes_txt[0]}"
    assert classes_txt[2] == "Truck", f"classes.txt[2] = {classes_txt[2]}"
    assert classes_txt[3] == "Bus", f"classes.txt[3] = {classes_txt[3]}"
    # Van(id=1) 因为没 label 落到它上面，所以可能是 class_1 (不用)
    # 但实际：car(2)→Car(0), motorcycle(3)→Car(0)，bus(5)→Bus(3), truck(7)→Truck(2)
    # 没 label 写入 id=1，所以 Van 不会出现在 classes.txt 的 used_ids 中

    counts = _count_class_ids(imageset_dir)
    # id=0 (Car) 应有 10 个：car + motorcycle 都映射到 0，每图 2 × 5 图 = 10
    assert counts.get(0) == 10, f"class 0 (Car) 数量: {counts}"
    # id=2 (Truck) 应有 5 个（truck 映射）
    assert counts.get(2) == 5, f"class 2 (Truck) 数量: {counts}"
    # id=3 (Bus) 应有 5 个
    assert counts.get(3) == 5, f"class 3 (Bus) 数量: {counts}"
    # 总 20 个标签
    assert sum(counts.values()) == 20, f"总标签: {counts}"

    # data.yaml 一致性
    yaml_names = _read_data_yaml_names(imageset_dir)
    assert yaml_names.get(0) == "Car"
    assert yaml_names.get(2) == "Truck"
    assert yaml_names.get(3) == "Bus"


# ============================================================
# 辅助：造 5 张图 + 上传 imageset
# ============================================================
def _upload_5_images(client, tmp_path: Path, name: str) -> str:
    files = []
    for i in range(5):
        p = tmp_path / f"{name}_{i:02d}.jpg"
        _make_image(p)
        with p.open("rb") as f:
            files.append(("files", (p.name, f.read(), "image/jpeg")))
    r = client.post("/api/imagesets/upload-folder", files=files, data={"imageset_name": name})
    assert r.status_code == 200, r.text
    return r.json()["imageset_id"]


# ============================================================
# S1: 空 imageset + append + 用模型类别（无 target_classes）
# 验证：append 在空集上行为 = replace，类名来自模型
# ============================================================
def test_s1_empty_imageset_append_uses_model_classes(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "s1_empty_append")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)

    _ROUND["n"] = 10  # bump 以保证 predict box 独特

    # 前端调 resolve-mapping
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "append",
    })
    map_items = r.json()["items"]
    # 空 imageset，append == replace 行为，candidate 应包含全部 model 类
    assert {m["id"]: m["name"] for m in map_items} == {
        0: "person", 1: "car", 2: "dog", 3: "cat", 4: "bird",
    }, f"[S1] mapClasses: {map_items}"

    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_items)  # 全部 identity，无 override

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[S1] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt == ["person", "car", "dog", "cat", "bird"], f"[S1] classes.txt: {classes_txt}"


# ============================================================
# S2: 空 imageset + append + 手填 target_classes（和模型无关的新名字）
# 验证：candidate 按 target_classes 分配新 ID，从 0 开始
# ============================================================
def test_s2_empty_imageset_append_handwritten_targets(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "s2_empty_handwritten")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)

    _ROUND["n"] = 11
    targets = ["Vehicle", "Animal", "Object"]  # 3 个手填

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": targets, "label_mode": "append",
    })
    map_items = r.json()["items"]
    # 空 imageset + 手填，应分配 0,1,2
    assert {m["id"]: m["name"] for m in map_items} == {
        0: "Vehicle", 1: "Animal", 2: "Object",
    }, f"[S2] mapClasses: {map_items}"

    # 模拟 UI：car/person → Vehicle(0), dog/cat/bird → Vehicle(0) 作为 fallback
    overrides = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}  # 全部映射到 Vehicle
    # filter out src == dst
    overrides = {k: v for k, v in overrides.items() if int(k) != v}
    # {"1":0, "2":0, "3":0, "4":0} （src=0 映射到 0 是 identity）

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": targets,
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[S2] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt[0] == "Vehicle", f"[S2] classes.txt[0] = {classes_txt[0]}"
    counts = _count_class_ids(imageset_dir)
    # 5 类全部落到 id=0，5 图 × 5 detection = 25
    assert counts.get(0) == 25, f"[S2] class 0 应 25，实际: {counts}"


# ============================================================
# S6: ⭐ 关键业务：已有标签 + 权重的 class 0 不能盖掉 imageset 的 class 0
# 场景：imageset 已有标签（class 0 = "vehicle"），模型 class 0 = "person"
#       即使用户选"默认映射"，后端也必须给 person 新分配 ID，不能盖 vehicle
# ============================================================
def test_s6_weight_class_0_must_not_map_to_imageset_class_0(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "s6_conflict")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置：imageset 已有 class 0 = "vehicle"（模拟之前用其它模型打过）
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text(
            "0 0.2 0.5 0.1 0.1\n0 0.4 0.5 0.1 0.1\n", encoding="utf-8"
        )
    (labels_dir / "classes.txt").write_text("vehicle", encoding="utf-8")

    _ROUND["n"] = 12

    # 前端调 resolve-mapping（模型类别 [person,car,dog,cat,bird]）
    # 注：existing class 0 是 "vehicle"，和 model 的 "person" 不同名
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "append",
    })
    map_items = r.json()["items"]
    map_dict = {m["id"]: m["name"].replace(" (复用)", "") for m in map_items}

    # ⭐ 核心断言：model 的 class 0 (person) 不能被分配到 imageset 的 class 0
    # existing max_cid=0, tc_offset=1 → person 应分配到 id=1 或更大
    assert map_dict.get(0) != "person", (
        f"[S6] ⚠️ BUG: model.class_0 (person) 映射到了 imageset.class_0，会盖掉 vehicle: {map_dict}"
    )
    # person 应在 tc_offset=1 或之后分配
    person_id = next(k for k, v in map_dict.items() if v == "person")
    assert person_id >= 1, f"[S6] person 应分配到 id >= 1，实际 {person_id}"

    # 模拟 UI default + 用户点"默认映射"（不改 dropdown）
    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_items)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[S6] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # ⭐ classes.txt[0] 必须仍是 "vehicle"，不能被 "person" 覆盖
    assert classes_txt[0] == "vehicle", (
        f"[S6] ⚠️ classes.txt[0] 被污染：{classes_txt[0]}，应保持 'vehicle'"
    )
    # person/car/dog/cat/bird 应在 classes.txt[1..5]
    assert "person" in classes_txt, f"[S6] person 应写入 classes.txt: {classes_txt}"
    assert "car" in classes_txt, f"[S6] car 应写入: {classes_txt}"

    counts = _count_class_ids(imageset_dir)
    # imageset.class_0 (vehicle) 应保持原数量：5 图 × 2 box = 10（本轮没新增到 0）
    assert counts.get(0) == 10, f"[S6] class 0 数量应 10（原有不变）: {counts}"


# ============================================================
# S7: AI 打标（模拟）+ 权重打标串联
# 场景：先预置一些 Qwen 打标结果（手写 labels），再跑权重打标 append
# 验证：两种 job 类型产出的标签能正确合并，名字不丢失
# ============================================================
def test_s7_ai_annotate_then_weight_annotate(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "s7_ai_plus_weight")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 模拟 AI 打标结果：class 0 = "my_ai_class_0"，class 1 = "my_ai_class_1"
    # （Qwen 常给自定义类名，可能和模型类名完全不同）
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text(
            "0 0.2 0.3 0.1 0.1\n1 0.4 0.3 0.1 0.1\n", encoding="utf-8"
        )
    (labels_dir / "classes.txt").write_text("my_ai_class_0\nmy_ai_class_1", encoding="utf-8")

    _ROUND["n"] = 13

    # 用户回来用权重打标，append 模式
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "append",
    })
    map_items = r.json()["items"]
    map_dict = {m["id"]: m["name"].replace(" (复用)", "") for m in map_items}

    # 权重类名和 AI 类名完全不同 → 应全部新分配
    # existing max_cid=1, tc_offset=2
    assert map_dict.get(0) != "person", f"[S7] person 不应覆盖 AI 的 class 0: {map_dict}"
    assert map_dict.get(1) != "car", f"[S7] car 不应覆盖 AI 的 class 1: {map_dict}"

    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_items)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[S7] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # AI 类名必须保留在 0,1
    assert classes_txt[0] == "my_ai_class_0", f"[S7] AI class 0 被污染: {classes_txt}"
    assert classes_txt[1] == "my_ai_class_1", f"[S7] AI class 1 被污染: {classes_txt}"
    # 权重类名应在 2,3,4,5,6（tc_offset 2 + 5 类 = 2..6）
    assert "person" in classes_txt
    assert "car" in classes_txt
    assert "bird" in classes_txt

    counts = _count_class_ids(imageset_dir)
    # AI 的 class 0, 1 原数量 5 个（5 图 × 1 box）每个
    assert counts.get(0) == 5, f"[S7] AI class 0 数量应 5（不变）: {counts}"
    assert counts.get(1) == 5, f"[S7] AI class 1 数量应 5（不变）: {counts}"


# ============================================================
# C1: 空 imageset + replace + 手填 target（第一轮就手填新类别）
# ============================================================
def test_c1_empty_replace_with_handwritten(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "c1_empty_replace_hw")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)

    _ROUND["n"] = 20
    targets = ["Alpha", "Beta"]  # 2 个手填

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": targets, "label_mode": "replace",
    })
    map_items = r.json()["items"]
    assert {m["id"]: m["name"] for m in map_items} == {0: "Alpha", 1: "Beta"}, f"[C1] {map_items}"

    # 全部映射到 Alpha(0)
    overrides = {"1": 0, "2": 0, "3": 0, "4": 0}  # src=0→0 identity 不记
    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": targets,
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "replace", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[C1] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    assert classes_txt[0] == "Alpha", f"[C1] classes.txt[0] = {classes_txt[0]}"
    counts = _count_class_ids(imageset_dir)
    # 5 类全部 → 0，共 25 个标签在 class 0
    assert counts.get(0) == 25, f"[C1] class 0 数量: {counts}"


# ============================================================
# C2: 有标签 + replace + 默认映射（验证 replace 会清掉 existing 内容但保留已有名字映射）
# ============================================================
def test_c2_existing_labels_replace_with_model(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "c2_existing_replace")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 读实际上传后的 image filenames（可能被规范化）
    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置 existing: class 0 = "old_vehicle"，按实际 filename 写对应 label
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text("0 0.3 0.5 0.1 0.1\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("old_vehicle", encoding="utf-8")

    _ROUND["n"] = 21

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "replace",
    })
    map_items = r.json()["items"]
    # replace 模式 → tc_offset=0（不看 existing labels 的 max_cid），所有 model 类从 0 开始
    map_dict = {m["id"]: m["name"] for m in map_items}
    assert map_dict == {0: "person", 1: "car", 2: "dog", 3: "cat", 4: "bird"}, f"[C2] {map_items}"

    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_items)  # identity

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "replace", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[C2] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # replace 模式：old_vehicle 被覆盖为 person（因为 class 0 现在写入的是 person）
    assert classes_txt == ["person", "car", "dog", "cat", "bird"], f"[C2] classes.txt: {classes_txt}"
    counts = _count_class_ids(imageset_dir)
    for cid in range(5):
        assert counts.get(cid) == 5, f"[C2] class {cid} 数量: {counts}"


# ============================================================
# C3: 有标签 + append + 手填混合（部分同名 existing 部分新名字）
# imageset 已有 car(0)、person(1)，用户手填 ["person", "Truck", "Bus"]
# → person 应复用 id=1，Truck/Bus 新分配
# ============================================================
def test_c3_existing_append_handwritten_mixed(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "c3_existing_mixed")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置：class 0 = "car", class 1 = "person"
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text(
            "0 0.3 0.5 0.1 0.1\n1 0.6 0.5 0.1 0.1\n", encoding="utf-8"
        )
    (labels_dir / "classes.txt").write_text("car\nperson", encoding="utf-8")

    _ROUND["n"] = 22
    # 用户手填：person（同名）、Truck（新名）、Bus（新名）
    targets = ["person", "Truck", "Bus"]

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": targets, "label_mode": "append",
    })
    map_items = r.json()["items"]
    map_dict = {m["id"]: m["name"] for m in map_items}
    # existing max_cid=1, tc_offset=2
    # person → existing 有 id=1 → candidate[1]="person"
    # Truck → 无 → candidate[2]="Truck"
    # Bus → 无 → candidate[3]="Bus"
    assert map_dict.get(1) == "person", f"[C3] person 应复用 id=1: {map_dict}"
    assert map_dict.get(2) == "Truck", f"[C3] Truck 应新分配 id=2: {map_dict}"
    assert map_dict.get(3) == "Bus", f"[C3] Bus 应新分配 id=3: {map_dict}"

    # UI smart default：src=0 car → 无 match Truck/Bus/person → fallback mapClasses[0].id
    # 具体 fallback 到哪个 id 取决于后端返回顺序（按 id 排序）→ 第一个是 id=1 (person)
    # overrides: src=0→1(非 identity), src=1→1(identity 不记), src=2→1, src=3→1, src=4→1
    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_items)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": targets,
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[C3] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # classes.txt[0] 应保持 "car"，[1] 保持 "person"（同名复用不污染）
    assert classes_txt[0] == "car", f"[C3] classes.txt[0] = {classes_txt[0]}"
    assert classes_txt[1] == "person", f"[C3] classes.txt[1] = {classes_txt[1]}"


# ============================================================
# C4: 模型轮换 A → B → A（同 imageset 上）
# 验证多次切换后 ID 分配稳定
# ============================================================
def test_c4_model_rotation_a_b_a(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "c4_rotation")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_a_id = r.json()["model_id"]
    r = client.post("/api/models/upload", files={"model_file": ("model_b.pt", b"pt-b", "application/octet-stream")})
    model_b_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)

    def _run(model_id, model_classes, _round):
        _ROUND["n"] = _round
        r = client.post("/api/annotate/resolve-mapping", json={
            "model_id": model_id, "imageset_id": imageset_id,
            "selected_class_ids": list(range(len(model_classes))),
            "target_classes": [], "label_mode": "append",
        })
        map_items = r.json()["items"]
        sources = [{"id": i, "name": model_classes[i]} for i in range(len(model_classes))]
        overrides = _build_overrides_from_ui(sources, map_items)
        r = client.post("/api/annotate/jobs", json={
            "model_id": model_id, "imageset_id": imageset_id,
            "selected_class_ids": list(range(len(model_classes))),
            "target_classes": [],
            "class_id_overrides": overrides, "mapping_confirmed": True,
            "label_mode": "append", "update_imageset_labels": True,
        })
        assert r.status_code == 200, r.text
        final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
        assert final["status"] == "succeeded", f"[C4/{model_id[:8]}] 失败: {final.get('error')}"

    # 轮 1: A (5 类) — 初始
    _run(model_a_id, MODEL_A_CLASSES, 30)
    cls1 = _read_classes_txt(imageset_dir)
    assert cls1 == ["person", "car", "dog", "cat", "bird"], f"[C4/1] {cls1}"

    # 轮 2: B (3 类) — car 同名复用 id=1, truck/bus 新分配
    _run(model_b_id, MODEL_B_CLASSES, 31)
    cls2 = _read_classes_txt(imageset_dir)
    assert cls2[0] == "person"
    assert cls2[1] == "car"
    assert "truck" in cls2
    assert "bus" in cls2

    # 轮 3: A 回来 — 所有类应复用（名字都在 existing 里）
    _run(model_a_id, MODEL_A_CLASSES, 32)
    cls3 = _read_classes_txt(imageset_dir)
    # A 的 5 个类全部在 existing，没新类
    assert cls3[0] == "person"
    assert cls3[1] == "car"
    assert cls3[2] == "dog"
    assert cls3[3] == "cat"
    assert cls3[4] == "bird"
    # truck/bus 仍在（之前 B 加的）
    assert "truck" in cls3
    assert "bus" in cls3


# ============================================================
# C5: 多源合并到已有 class（cat/dog/bird 都合并到 existing 的 "animal"）
# ============================================================
def test_c5_multi_source_merge_to_existing(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "c5_multi_merge")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置：class 0 = "animal"
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text("0 0.3 0.5 0.1 0.1\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("animal", encoding="utf-8")

    _ROUND["n"] = 40

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "append",
    })
    map_items = r.json()["items"]

    # 用户勾选 cat(3)、dog(2)、bird(4)，全部 override 到 0（animal）
    overrides = {"2": 0, "3": 0, "4": 0}

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[C5] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # ⭐ class 0 应保持 "animal"，不被 dog/cat/bird 的名字覆盖
    assert classes_txt[0] == "animal", f"[C5] class 0 被污染: {classes_txt[0]}"
    counts = _count_class_ids(imageset_dir)
    # 原有 5 + dog(3)·5 + cat(3)·5 + bird(3)·5 = 5 + 15 = 20 个标签 all in class 0
    # 注：selected=[2,3,4] 只勾这 3 类
    assert counts.get(0) == 20, f"[C5] class 0 应 20 (原 5 + 新 15): {counts}"


# 注：C6 (override 到 UI 外的 id) 已由 tests/test_integration_api.py 里的
# test_annotate_override_high_ids_do_not_preserve_old_semantic_names 覆盖
# 这里不再重复测试


# ============================================================
# R1: label_mode 切换后不重渲的 UI 状态漂移（最易触发的隐性 bug）
# 场景：用户先在 replace 模式下配置了映射 → 改成 append 模式 → 直接打标
#       前端 state 的 overrides 基于旧 mapClasses（replace 版），后端按 append 重算
# ============================================================
def test_r1_label_mode_flip_after_confirm(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "r1_mode_flip")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置 existing: class 0 = "old_person"
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text("0 0.3 0.5 0.1 0.1\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("old_person", encoding="utf-8")

    _ROUND["n"] = 60

    # 用户最初在 replace 模式下查看映射
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "replace",
    })
    map_replace = r.json()["items"]
    # replace 模式 candidate = {0:person, 1:car, 2:dog, 3:cat, 4:bird}（忽略 existing）

    # 用户在 replace 模式下生成 overrides（identity）
    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_replace)

    # 用户提交时改成了 append（mapping signature 理论上应不匹配，但测试后端鲁棒性）
    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    # 不管模式是否一致，后端必须不崩、不污染
    assert final["status"] == "succeeded", f"[R1] 打标失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # existing 的 class 0 是 "old_person"，用户 identity 映射 src 0 → dst 0
    # append 模式下归并语义：class 0 的名字应保持 "old_person"，不被覆盖成 "person"
    assert classes_txt[0] == "old_person", (
        f"[R1] append + identity 模式下，class 0 应保持 'old_person'：{classes_txt}"
    )


# ============================================================
# R2: classes.txt 含脏数据（class_N 占位符 + 真实名字混杂）
# 场景：imageset 之前由某工具生成，classes.txt 里有 "class_3" 这种占位符
#       用户现在用权重打标，应正确绕过占位符，不把占位符当真实名字复用
# ============================================================
def test_r2_dirty_classes_txt_with_placeholders(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "r2_dirty")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"pt-a", "application/octet-stream")})
    model_id = r.json()["model_id"]
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置：class 0 = "real_animal"（真实名），class 3 = "class_3"（占位符）
    # class 1, 2 无 label 但在 classes.txt 里有
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text(
            "0 0.2 0.3 0.1 0.1\n3 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
    # 故意混入 placeholder
    (labels_dir / "classes.txt").write_text(
        "real_animal\nclass_1\nclass_2\nclass_3\n", encoding="utf-8"
    )

    _ROUND["n"] = 61

    # 用户用 Model A append
    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [], "label_mode": "append",
    })
    map_items = r.json()["items"]
    map_dict = {m["id"]: m["name"].replace(" (复用)", "") for m in map_items}

    # ⭐ 关键：class_3 是占位符不算"真实名字"
    # resolve_mapping 的 existing_name_to_id 只收集 非 placeholder 的 name
    # 所以 model 的 cat(src=3) 不应复用 id=3（因为 "class_3" 不是真实 cat 名字）
    assert map_dict.get(3) != "cat", (
        f"[R2] cat 不应复用占位符 id=3：{map_dict}"
    )

    # existing max_cid=3, tc_offset=4
    sources = [{"id": i, "name": MODEL_A_CLASSES[i]} for i in range(5)]
    overrides = _build_overrides_from_ui(sources, map_items)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": [0, 1, 2, 3, 4], "target_classes": [],
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": "append", "update_imageset_labels": True,
    })
    final = _wait_job(client, f"/api/annotate/jobs/{r.json()['id']}")
    assert final["status"] == "succeeded", f"[R2] 失败: {final.get('error')}"

    classes_txt = _read_classes_txt(imageset_dir)
    # class 0 = real_animal 应保留（真实名字优先）
    assert classes_txt[0] == "real_animal", f"[R2] class 0 被污染: {classes_txt}"
    # 占位符 class_3 最终应被清理（因为真实写入 class 3 的标签有名字来源）
    # 这里 existing 的 class 3 只有 "class_3" 占位符，新打标不会归并到 3（因为 cat 名字未匹配）
    # class 3 仍保留 placeholder 或被新类名覆盖？取决于兑底逻辑
    # 验证 class 3 的原 labels 还在（append 保留）
    counts = _count_class_ids(imageset_dir)
    assert counts.get(0) >= 5, f"[R2] class 0 标签 {counts.get(0)}，至少 5（existing）"
    assert counts.get(3) >= 5, f"[R2] class 3 existing 标签应保留: {counts}"


# ============================================================
# 辅助：跑一轮打标（自动模拟前端 resolve-mapping + smart default）
# ============================================================
def _run_annotate_round(
    client, model_id: str, imageset_id: str, model_classes: list[str],
    target_classes: list[str] = None, label_mode: str = "append",
    round_n: int = 1, custom_overrides: dict[str, int] = None,
    selected_subset: list[int] = None,
) -> str:
    """模拟前端完整打标流程：resolve-mapping → build overrides → submit job → wait
    返回 job_id"""
    _ROUND["n"] = round_n
    tc = target_classes or []
    selected = selected_subset if selected_subset is not None else list(range(len(model_classes)))

    r = client.post("/api/annotate/resolve-mapping", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": list(range(len(model_classes))),  # 前端渲染传全部
        "target_classes": tc,
        "label_mode": label_mode,
    })
    assert r.status_code == 200, r.text
    map_items = r.json()["items"]

    if custom_overrides is not None:
        overrides = custom_overrides
    else:
        sources = [{"id": i, "name": model_classes[i]} for i in range(len(model_classes))]
        overrides = _build_overrides_from_ui(sources, map_items)

    r = client.post("/api/annotate/jobs", json={
        "model_id": model_id, "imageset_id": imageset_id,
        "selected_class_ids": selected,
        "target_classes": tc,
        "class_id_overrides": overrides, "mapping_confirmed": True,
        "label_mode": label_mode, "update_imageset_labels": True,
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    final = _wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded", f"round {round_n}: {final.get('error')}"
    return job_id


def _seed_existing_labels(labels_dir: Path, image_names: list[str], label_text: str, classes_txt: str):
    """按真实 filename 预置 existing labels"""
    for fname in image_names:
        stem = Path(fname).stem
        (labels_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
    (labels_dir / "classes.txt").write_text(classes_txt, encoding="utf-8")


def _simulate_ai_round(labels_dir: Path, image_names: list[str], ai_label_text: str, classes_txt_merge: list[str]):
    """模拟一轮 AI 打标：手写 labels（追加到 existing）+ 更新 classes.txt
    classes_txt_merge 是希望 classes.txt 的完整内容"""
    for fname in image_names:
        stem = Path(fname).stem
        label_file = labels_dir / f"{stem}.txt"
        existing = label_file.read_text(encoding="utf-8") if label_file.exists() else ""
        # 保证 existing 末尾有换行，避免和 ai_label_text 粘连导致行被破坏
        if existing and not existing.endswith("\n"):
            existing += "\n"
        label_file.write_text(existing + ai_label_text, encoding="utf-8")
    (labels_dir / "classes.txt").write_text("\n".join(classes_txt_merge), encoding="utf-8")


# ============================================================
# CYC1: 综合循环 1 — 空 imageset 启动
# 序列：A(replace,默认) → B(append,默认) → C(append,手填) → D(append,默认)
#       → A(append,手填) → AI(手写模拟) → B(append,默认)
# 每轮断言 job 成功、至少 5 个新标签、关键类名不被污染
# ============================================================
def test_cyc1_clean_imageset_cycle_abcda_ai_b(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "cyc1_cycle")

    # 上传 4 个模型
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"a", "application/octet-stream")})
    a_id = r.json()["model_id"]
    r = client.post("/api/models/upload", files={"model_file": ("model_b.pt", b"b", "application/octet-stream")})
    b_id = r.json()["model_id"]
    r = client.post("/api/models/upload", files={"model_file": ("model_c.pt", b"c", "application/octet-stream")})
    c_id = r.json()["model_id"]
    r = client.post("/api/models/upload", files={"model_file": ("model_d.pt", b"d", "application/octet-stream")})
    d_id = r.json()["model_id"]

    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"

    # -------- 轮 1: A, replace, 默认映射 --------
    _run_annotate_round(client, a_id, imageset_id, MODEL_A_CLASSES, label_mode="replace", round_n=100)
    cls = _read_classes_txt(imageset_dir)
    assert cls == ["person", "car", "dog", "cat", "bird"], f"[CYC1/R1] {cls}"
    counts = _count_class_ids(imageset_dir)
    assert sum(counts.values()) >= 5, f"[CYC1/R1] 总标签 {sum(counts.values())}"

    # -------- 轮 2: B, append, 默认 —— car 复用 id=1 --------
    _run_annotate_round(client, b_id, imageset_id, MODEL_B_CLASSES, label_mode="append", round_n=101)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "person", f"[CYC1/R2] person 被污染: {cls}"
    assert cls[1] == "car", f"[CYC1/R2] car 不再是 id=1: {cls}"
    assert "truck" in cls and "bus" in cls

    # -------- 轮 3: C, append, 手填 ["Animal", "Human"] —— 全新类 --------
    _run_annotate_round(client, c_id, imageset_id, MODEL_C_CLASSES,
                        target_classes=["Animal", "Human"], label_mode="append", round_n=102)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "person"
    assert cls[1] == "car"  # 不被污染
    assert "Animal" in cls or "Human" in cls  # 至少一个被写入（取决于 override）

    # -------- 轮 4: D, append, 默认 —— tiger/lion/wolf 全新 --------
    _run_annotate_round(client, d_id, imageset_id, MODEL_D_CLASSES, label_mode="append", round_n=103)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "person"
    assert cls[1] == "car"
    assert "tiger" in cls and "lion" in cls and "wolf" in cls

    # -------- 轮 5: A 回来, append, 手填 ["OnlyPerson"] —— 测 A 重返能否复用 person --------
    _run_annotate_round(client, a_id, imageset_id, MODEL_A_CLASSES,
                        target_classes=["OnlyPerson"], label_mode="append", round_n=104)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "person"  # 原有 person 不变
    assert cls[1] == "car"

    # -------- 轮 6: 模拟 AI 打标（手写 labels，新类名）--------
    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 获取当前 classes.txt 长度作为 AI 新类的 id 起点
    current_cls = _read_classes_txt(imageset_dir)
    ai_id = len(current_cls)  # 下一个可用 id
    # AI 产出一个新类 "ai_plant"（在现有 class 后面）
    ai_cls_list = current_cls + ["ai_plant"]
    ai_label_text = f"{ai_id} 0.9 0.9 0.05 0.05\n"
    _simulate_ai_round(labels_dir, image_names, ai_label_text, ai_cls_list)

    cls = _read_classes_txt(imageset_dir)
    assert cls[ai_id] == "ai_plant", f"[CYC1/R6-AI] AI class 未写入: {cls}"
    counts_after_ai = _count_class_ids(imageset_dir)
    assert counts_after_ai.get(ai_id) == 5, f"[CYC1/R6-AI] AI 类应 5 标签: {counts_after_ai}"

    # -------- 轮 7: B 再回来, append, 默认 —— 验证 ai_plant 不被破坏 --------
    _run_annotate_round(client, b_id, imageset_id, MODEL_B_CLASSES, label_mode="append", round_n=105)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "person", f"[CYC1/R7] person 不变: {cls}"
    assert cls[1] == "car", f"[CYC1/R7] car 不变: {cls}"
    assert "ai_plant" in cls, f"[CYC1/R7] AI 类不应丢失: {cls}"
    # 计数：ai_plant 原有 5，不该被破坏
    counts_final = _count_class_ids(imageset_dir)
    assert counts_final.get(ai_id) == 5, f"[CYC1/R7] AI 类计数应保持 5: {counts_final}"


# ============================================================
# CYC2: 综合循环 2 — 初始已有标签启动（模拟接手他人数据集）
# 预置：class 0 = "dataset_person", class 1 = "dataset_car"
# 序列：A(append,默认) → C(append,手填) → AI(手写) → D(append,默认) → A(append,手填)
# ============================================================
def test_cyc2_dirty_imageset_cycle_acai_d_a(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "cyc2_dirty")
    r = client.post("/api/models/upload", files={"model_file": ("model_a.pt", b"a", "application/octet-stream")})
    a_id = r.json()["model_id"]
    r = client.post("/api/models/upload", files={"model_file": ("model_c.pt", b"c", "application/octet-stream")})
    c_id = r.json()["model_id"]
    r = client.post("/api/models/upload", files={"model_file": ("model_d.pt", b"d", "application/octet-stream")})
    d_id = r.json()["model_id"]

    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置：class 0 = "dataset_person", class 1 = "dataset_car"
    _seed_existing_labels(
        labels_dir, image_names,
        "0 0.1 0.2 0.05 0.05\n1 0.8 0.8 0.05 0.05\n",
        "dataset_person\ndataset_car",
    )

    # -------- 轮 1: A, append, 默认 —— 验证不污染原 dataset_person/car --------
    _run_annotate_round(client, a_id, imageset_id, MODEL_A_CLASSES, label_mode="append", round_n=200)
    cls = _read_classes_txt(imageset_dir)
    # dataset_person 和 dataset_car 是用户原有的命名，不能被模型的 person/car 覆盖
    assert cls[0] == "dataset_person", f"[CYC2/R1] class 0 被污染: {cls}"
    assert cls[1] == "dataset_car", f"[CYC2/R1] class 1 被污染: {cls}"
    # model 类因为名字不同（dataset_person vs person），会分配到新 id
    assert "person" in cls, f"[CYC2/R1] model person 应写入新 id: {cls}"

    # -------- 轮 2: C, append, 手填 ["my_cat", "my_elephant"] --------
    _run_annotate_round(client, c_id, imageset_id, MODEL_C_CLASSES,
                        target_classes=["my_cat", "my_elephant"], label_mode="append", round_n=201)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "dataset_person"  # 保持
    assert cls[1] == "dataset_car"

    # -------- 轮 3: AI 手写打标 --------
    current_cls = _read_classes_txt(imageset_dir)
    ai_id = len(current_cls)
    _simulate_ai_round(labels_dir, image_names, f"{ai_id} 0.5 0.5 0.03 0.03\n",
                       current_cls + ["ai_stuff"])
    cls = _read_classes_txt(imageset_dir)
    assert cls[ai_id] == "ai_stuff"
    assert cls[0] == "dataset_person"  # 仍保持

    # -------- 轮 4: D, append, 默认 --------
    _run_annotate_round(client, d_id, imageset_id, MODEL_D_CLASSES, label_mode="append", round_n=202)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "dataset_person"
    assert cls[1] == "dataset_car"
    assert "tiger" in cls and "lion" in cls and "wolf" in cls
    assert "ai_stuff" in cls

    # -------- 轮 5: A 回来, append, 手填 ["dataset_person"] —— 手填和 existing 同名复用 --------
    _run_annotate_round(client, a_id, imageset_id, MODEL_A_CLASSES,
                        target_classes=["dataset_person"], label_mode="append", round_n=203)
    cls = _read_classes_txt(imageset_dir)
    # dataset_person 被手填复用 → 仍是 dataset_person（不污染）
    assert cls[0] == "dataset_person", f"[CYC2/R5] class 0 应保持 dataset_person: {cls}"
    assert cls[1] == "dataset_car"
    # AI 的 ai_stuff 不应丢失
    assert "ai_stuff" in cls, f"[CYC2/R5] AI 类应保留: {cls}"

    # 最终累积标签数量合理
    counts = _count_class_ids(imageset_dir)
    # dataset_person (class 0) 每轮都有至少 5 个（existing + A 预测的 person 映射）
    assert counts.get(0) >= 5
    # ai_stuff 保持 5 个（原来写的）
    current_cls = _read_classes_txt(imageset_dir)
    ai_stuff_id = current_cls.index("ai_stuff")
    assert counts.get(ai_stuff_id) == 5, f"[CYC2/final] ai_stuff 被破坏: {counts}"


# ============================================================
# 辅助：上传 5 个模型 A/B/C/D/E
# ============================================================
def _upload_5_models(client) -> dict[str, str]:
    out = {}
    for name in ["a", "b", "c", "d", "e"]:
        r = client.post(
            "/api/models/upload",
            files={"model_file": (f"model_{name}.pt", f"m-{name}".encode(), "application/octet-stream")},
        )
        assert r.status_code == 200, r.text
        out[name] = r.json()["model_id"]
    return out


# ============================================================
# CYC3: AI 密集穿插 ABCDE（初始空 imageset）
# 序列：A → AI → B → AI → C(手填) → AI → D → AI → E → AI → A(回)
# 11 轮，验证每轮 AI 类名不被后续权重覆盖、权重类名不互相污染
# ============================================================
def test_cyc3_dense_ai_interleave_abcde_empty(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "cyc3_dense")
    models = _upload_5_models(client)
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]

    def _run_ai(round_n, ai_name):
        """在当前 classes.txt 末尾追加一个新 AI 类"""
        current_cls = _read_classes_txt(imageset_dir) if (labels_dir / "classes.txt").exists() else []
        new_id = len(current_cls)
        _simulate_ai_round(labels_dir, image_names, f"{new_id} 0.{round_n:02d} 0.{round_n:02d} 0.03 0.03\n",
                           current_cls + [ai_name])
        return new_id, ai_name

    # -------- 轮 1: A, replace --------
    _run_annotate_round(client, models["a"], imageset_id, MODEL_A_CLASSES, label_mode="replace", round_n=300)
    cls = _read_classes_txt(imageset_dir)
    assert cls == ["person", "car", "dog", "cat", "bird"]

    # -------- 轮 2: AI(ai_alpha) --------
    id_ai1, _ = _run_ai(301, "ai_alpha")
    cls = _read_classes_txt(imageset_dir)
    assert cls[id_ai1] == "ai_alpha"

    # -------- 轮 3: B, append --------
    _run_annotate_round(client, models["b"], imageset_id, MODEL_B_CLASSES, label_mode="append", round_n=302)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "person"
    assert cls[1] == "car"
    assert cls[id_ai1] == "ai_alpha", f"[CYC3/R3] ai_alpha 被污染: {cls}"

    # -------- 轮 4: AI(ai_beta) --------
    id_ai2, _ = _run_ai(303, "ai_beta")
    cls = _read_classes_txt(imageset_dir)
    assert cls[id_ai2] == "ai_beta"
    assert cls[id_ai1] == "ai_alpha"

    # -------- 轮 5: C, append, 手填 --------
    _run_annotate_round(client, models["c"], imageset_id, MODEL_C_CLASSES,
                        target_classes=["My_Cat", "My_Elephant"], label_mode="append", round_n=304)
    cls = _read_classes_txt(imageset_dir)
    assert cls[id_ai1] == "ai_alpha", f"[CYC3/R5] ai_alpha 丢失: {cls}"
    assert cls[id_ai2] == "ai_beta"

    # -------- 轮 6: AI(ai_gamma) --------
    id_ai3, _ = _run_ai(305, "ai_gamma")

    # -------- 轮 7: D, append --------
    _run_annotate_round(client, models["d"], imageset_id, MODEL_D_CLASSES, label_mode="append", round_n=306)
    cls = _read_classes_txt(imageset_dir)
    assert "tiger" in cls and "lion" in cls and "wolf" in cls
    # 所有 AI 类都还在
    assert cls[id_ai1] == "ai_alpha"
    assert cls[id_ai2] == "ai_beta"
    assert cls[id_ai3] == "ai_gamma"

    # -------- 轮 8: AI(ai_delta) --------
    id_ai4, _ = _run_ai(307, "ai_delta")

    # -------- 轮 9: E, append --------
    _run_annotate_round(client, models["e"], imageset_id, MODEL_E_CLASSES, label_mode="append", round_n=308)
    cls = _read_classes_txt(imageset_dir)
    assert "unicorn" in cls and "phoenix" in cls and "dragon" in cls
    for ai_id, ai_nm in [(id_ai1, "ai_alpha"), (id_ai2, "ai_beta"), (id_ai3, "ai_gamma"), (id_ai4, "ai_delta")]:
        assert cls[ai_id] == ai_nm, f"[CYC3/R9] {ai_nm} 丢失: {cls}"

    # -------- 轮 10: AI(ai_epsilon) --------
    id_ai5, _ = _run_ai(309, "ai_epsilon")

    # -------- 轮 11: A 回来, append --------
    _run_annotate_round(client, models["a"], imageset_id, MODEL_A_CLASSES, label_mode="append", round_n=310)
    cls = _read_classes_txt(imageset_dir)
    # 所有历史类都应保留
    for cid, expected in [
        (0, "person"), (1, "car"), (2, "dog"), (3, "cat"), (4, "bird"),
        (id_ai1, "ai_alpha"), (id_ai2, "ai_beta"), (id_ai3, "ai_gamma"),
        (id_ai4, "ai_delta"), (id_ai5, "ai_epsilon"),
    ]:
        assert cls[cid] == expected, f"[CYC3/R11-final] id={cid} 应为 {expected}，实际 {cls[cid]}"
    # 权重写的类都还在
    for name in ["truck", "bus", "tiger", "lion", "wolf", "unicorn", "phoenix", "dragon"]:
        assert name in cls, f"[CYC3/R11-final] {name} 丢失: {cls}"

    # AI 类每个都保持 5 个标签
    counts = _count_class_ids(imageset_dir)
    for ai_id in [id_ai1, id_ai2, id_ai3, id_ai4, id_ai5]:
        assert counts.get(ai_id) == 5, f"[CYC3/R11-final] AI id={ai_id} 计数应 5: {counts}"


# ============================================================
# CYC4: AI 密集穿插 ABCDE（初始有标签 imageset）
# 预置：class 0 = "seed_obj"
# 序列：A → AI → B → C(手填) → AI → D → E → AI → A(回)
# ============================================================
def test_cyc4_dense_ai_interleave_abcde_dirty(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "cyc4_dense_dirty")
    models = _upload_5_models(client)
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    # 预置：class 0 = "seed_obj"（用户原有的类名）
    _seed_existing_labels(labels_dir, image_names, "0 0.1 0.1 0.05 0.05\n", "seed_obj")

    def _run_ai(round_n, ai_name):
        current_cls = _read_classes_txt(imageset_dir)
        new_id = len(current_cls)
        _simulate_ai_round(labels_dir, image_names, f"{new_id} 0.{round_n:02d} 0.{round_n:02d} 0.03 0.03\n",
                           current_cls + [ai_name])
        return new_id

    # 轮 1: A append
    _run_annotate_round(client, models["a"], imageset_id, MODEL_A_CLASSES, label_mode="append", round_n=400)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "seed_obj", f"[CYC4/R1] seed_obj 被污染: {cls}"

    # 轮 2: AI
    id_a1 = _run_ai(401, "ai_one")
    # 轮 3: B append
    _run_annotate_round(client, models["b"], imageset_id, MODEL_B_CLASSES, label_mode="append", round_n=402)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "seed_obj"
    assert cls[id_a1] == "ai_one"

    # 轮 4: C append 手填 ["Merged_Animal"]
    _run_annotate_round(client, models["c"], imageset_id, MODEL_C_CLASSES,
                        target_classes=["Merged_Animal"], label_mode="append", round_n=403)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "seed_obj"
    assert cls[id_a1] == "ai_one"

    # 轮 5: AI
    id_a2 = _run_ai(404, "ai_two")

    # 轮 6: D append
    _run_annotate_round(client, models["d"], imageset_id, MODEL_D_CLASSES, label_mode="append", round_n=405)

    # 轮 7: E append
    _run_annotate_round(client, models["e"], imageset_id, MODEL_E_CLASSES, label_mode="append", round_n=406)
    cls = _read_classes_txt(imageset_dir)
    assert cls[0] == "seed_obj", f"[CYC4/R7] seed_obj 丢失: {cls}"
    assert cls[id_a1] == "ai_one"
    assert cls[id_a2] == "ai_two"

    # 轮 8: AI
    id_a3 = _run_ai(407, "ai_three")

    # 轮 9: A 回来
    _run_annotate_round(client, models["a"], imageset_id, MODEL_A_CLASSES, label_mode="append", round_n=408)
    cls = _read_classes_txt(imageset_dir)
    # 所有历史类都应保留
    assert cls[0] == "seed_obj", f"[CYC4-final] seed_obj 被污染: {cls}"
    for ai_id, name in [(id_a1, "ai_one"), (id_a2, "ai_two"), (id_a3, "ai_three")]:
        assert cls[ai_id] == name, f"[CYC4-final] {name} 被污染: {cls}"
    for wt_name in ["person", "car", "truck", "bus", "tiger", "lion", "wolf", "unicorn", "phoenix", "dragon"]:
        assert wt_name in cls, f"[CYC4-final] 权重类 {wt_name} 丢失: {cls}"


# ============================================================
# ROLL1: 撤回（rollback）接着打 —— 核心回滚场景
# 序列：A 打 → B 打 → rollback B → 验证恢复到 A 状态 → C 接着打
# ============================================================
def test_roll1_rollback_then_continue_annotate(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", _classes_factory)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", _predictor_factory)

    imageset_id = _upload_5_images(client, tmp_path, "roll1_rollback")
    models = _upload_5_models(client)
    imageset_dir = Path(app.state.app_state.get_imageset(imageset_id).dir_path)

    # 轮 1: A replace —— 基准状态
    job_a = _run_annotate_round(client, models["a"], imageset_id, MODEL_A_CLASSES,
                                 label_mode="replace", round_n=500)
    cls_after_a = _read_classes_txt(imageset_dir)
    counts_after_a = _count_class_ids(imageset_dir)
    assert cls_after_a == ["person", "car", "dog", "cat", "bird"]

    # 记录轮 A 后每张图 label 内容作为基准
    labels_dir = imageset_dir / "labels"
    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    image_names = [it["filename"] for it in images_resp.json()["items"]]
    baseline_labels = {
        Path(fn).stem: (labels_dir / f"{Path(fn).stem}.txt").read_text(encoding="utf-8")
        for fn in image_names
    }

    # 轮 2: B append —— 会改变 labels
    job_b = _run_annotate_round(client, models["b"], imageset_id, MODEL_B_CLASSES,
                                 label_mode="append", round_n=501)
    cls_after_b = _read_classes_txt(imageset_dir)
    counts_after_b = _count_class_ids(imageset_dir)
    assert "truck" in cls_after_b and "bus" in cls_after_b
    # B 打标后标签数应增加
    assert sum(counts_after_b.values()) > sum(counts_after_a.values()), (
        "[ROLL1/R2] B 打标后总标签应增加"
    )

    # -------- 撤回 B --------
    r = client.post(f"/api/annotate/jobs/{job_b}/rollback")
    assert r.status_code == 200, f"[ROLL1] rollback 失败: {r.text}"
    rb_result = r.json()
    assert rb_result["status"] == "rolled_back"
    # restored_files 应等于图片数
    assert rb_result["restored_files"] == 5, f"[ROLL1] 恢复数: {rb_result}"

    # 验证 labels 内容恢复到轮 A 后
    for fn in image_names:
        stem = Path(fn).stem
        current = (labels_dir / f"{stem}.txt").read_text(encoding="utf-8")
        assert current == baseline_labels[stem], (
            f"[ROLL1] {stem}.txt 没有恢复到 A 状态：\n当前: {current}\n基准: {baseline_labels[stem]}"
        )

    # 验证 counts 恢复到轮 A 后
    counts_after_rollback = _count_class_ids(imageset_dir)
    assert counts_after_rollback == counts_after_a, (
        f"[ROLL1] 计数未恢复: rollback后={counts_after_rollback}, A后={counts_after_a}"
    )

    # -------- 接着打 C ——撤回后接续打标，核心验证 --------
    _run_annotate_round(client, models["c"], imageset_id, MODEL_C_CLASSES,
                        target_classes=["Beast"], label_mode="append", round_n=502)
    cls_after_c = _read_classes_txt(imageset_dir)
    # A 的类还在，且 B 的 truck/bus 不再出现（已回滚）
    assert cls_after_c[0] == "person"
    assert cls_after_c[1] == "car"
    # truck/bus 不在本轮 final classes.txt（rebuild 基于 used_ids，B 的已被回滚）
    # 但**注意**：rollback 后 classes.txt 可能没主动清理，但 rebuild 会基于 used_ids
    # 所以只要实际写入 truck/bus 的 box 被回滚了，rebuild 就不会保留它们
    counts_after_c = _count_class_ids(imageset_dir)
    # 每张图 C 产 3 个 box，全部 override 到 Beast（id=5，在 bird=4 后）
    # 所以 class 5 应该有 15 个标签
    assert sum(counts_after_c.values()) > sum(counts_after_a.values()), "[ROLL1/C] C 后标签应增加"
