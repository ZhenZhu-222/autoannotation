# ============================================================
# Qwen 大模型打标路由 (Controller)
# 职责：通过 Qwen 视觉大模型进行自动标注（支持文字描述 + 参考图片）
# 规范：仅参数接收与异常转换，逻辑委托 service
# ============================================================
from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_login
from app.core.rbac import can_operate_imageset, can_view_job
from app.core.state import AppState
from app.schemas.common import JobStatusResponse, JobSubmitResponse
from app.services.qwen_annotation_service import QwenAnnotationService
from app.services.qwen_upload_service import QwenUploadService
from app.services.task_manager import TaskManager

router = APIRouter(prefix="/api", tags=["qwen-annotate"])


# 创建 Qwen VLM 零样本打标任务（传入文字描述+参考图片，AI 自动识别并生成标注框写入图片集）
@router.post("/qwen/annotate/jobs")
async def create_qwen_annotate_job(
    imageset_id: str = Form(...),
    description: str = Form(default=""),
    qwen_model: str = Form(default=""),
    api_key: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] = File(default=[]),
    files_bracket: list[UploadFile] = File(default=[], alias="files[]"),
    qwen_ref_images: list[UploadFile] = File(default=[], alias="qwenRefImages"),
    label_task: str = Form(default="detect"),
    label_mode: str = Form(default="append"),
    update_imageset_labels: str = Form(default="true"),
    round_tag: str = Form(default=""),
    operator: str = Form(default="anonymous"),
    save_overlays: str = Form(default="true"),
    strict_size_check: str = Form(default="true"),
    size_retry: int = Form(default=1),
    precision_mode: str = Form(default="strict"),
    sample_count: int = Form(default=3),
    max_calibration_error_px: float = Form(default=8.0),
    min_consensus_rate: float = Form(default=0.67),
    duplicate_iou: float = Form(default=0.98),
    reject_on_invalid_svg: str = Form(default="true"),
    green_hsv_profile: str = Form(default="neon"),
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobSubmitResponse:
    imageset = state.get_imageset(imageset_id)
    if not imageset:
        raise HTTPException(status_code=404, detail="imageset_id 不存在")
    if not can_operate_imageset(current_user, imageset):
        raise HTTPException(status_code=403, detail="无权操作该图片集")

    # ---------- 收集上传文件 ----------
    all_files: list[UploadFile] = []
    if file is not None:
        all_files.append(file)
    all_files.extend(files or [])
    all_files.extend(files_bracket or [])
    all_files.extend(qwen_ref_images or [])

    desc = (description or "").strip()
    if not desc and not all_files:
        raise HTTPException(status_code=400, detail="请至少提供文字描述或参考图片")

    # ---------- 参数校验（委托 service） ----------
    try:
        params = QwenUploadService.validate_and_build_params(
            label_mode=label_mode, update_imageset_labels=update_imageset_labels,
            save_overlays=save_overlays, strict_size_check=strict_size_check,
            precision_mode=precision_mode, sample_count=sample_count,
            max_calibration_error_px=max_calibration_error_px,
            min_consensus_rate=min_consensus_rate, duplicate_iou=duplicate_iou,
            reject_on_invalid_svg=reject_on_invalid_svg, green_hsv_profile=green_hsv_profile,
            size_retry=size_retry, label_task=label_task,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # ---------- 参考图暂存（委托 service） ----------
    upload_dir, ref_paths, invalid_refs = await QwenUploadService.stage_ref_images(all_files)

    if not desc and not ref_paths:
        shutil.rmtree(upload_dir, ignore_errors=True)
        if invalid_refs:
            bad = ", ".join(invalid_refs[:3])
            raise HTTPException(status_code=400, detail=f"参考图片解析失败: {bad}。请上传 jpg/png/webp/bmp。")
        raise HTTPException(status_code=400, detail="请至少提供文字描述或参考图片")

    # ---------- 构建 payload ----------
    clean_operator = current_user.username
    clean_round_tag = (round_tag or "").strip()
    clean_qwen_model = (qwen_model or "").strip()
    clean_api_key = (api_key or "").strip()

    payload = {
        "imageset_id": imageset_id,
        "description": desc,
        "qwen_model": clean_qwen_model,
        "label_task": params["label_task"],
        "label_mode": label_mode,
        "round_tag": clean_round_tag,
        "operator": clean_operator,
        "reference_count": len(ref_paths),
        "invalid_reference_count": len(invalid_refs),
        "api_key_provided": bool(clean_api_key),
        **params,
    }

    def runner(ctx):
        try:
            return QwenAnnotationService.run_qwen_annotate_job(
                state=state, job_ctx=ctx,
                imageset_id=imageset_id, description=desc,
                reference_files=ref_paths,
                qwen_model=clean_qwen_model, api_key=clean_api_key,
                label_task=params["label_task"],
                label_mode=label_mode,
                update_imageset_labels=params["update_imageset_labels"],
                round_tag=clean_round_tag, operator=clean_operator,
                save_overlays=params["save_overlays"],
                strict_size_check=params["strict_size_check"],
                size_retry=params["size_retry"],
                precision_mode=params["precision_mode"],
                sample_count=params["sample_count"],
                max_calibration_error_px=params["max_calibration_error_px"],
                min_consensus_rate=params["min_consensus_rate"],
                duplicate_iou=params["duplicate_iou"],
                reject_on_invalid_svg=params["reject_on_invalid_svg"],
                green_hsv_profile=params["green_hsv_profile"],
            )
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    job = tasks.submit(
        job_type="qwen_annotate",
        payload=payload,
        runner=runner,
        operator_id=current_user.id,
        operator_username=current_user.username,
    )
    return {"id": job.id, "job_type": job.job_type, "status": job.status, "payload": job.payload, "created_at": job.created_at}


# 查询 Qwen 打标任务实时状态（进度、是否完成、错误信息）
@router.get("/qwen/annotate/jobs/{job_id}")
def get_qwen_annotate_job(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobStatusResponse:
    job = tasks.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    if job.job_type != "qwen_annotate":
        raise HTTPException(status_code=400, detail="job 类型不是 qwen_annotate")
    if not can_view_job(current_user, job):
        raise HTTPException(status_code=403, detail="无权查看该任务")
    return job.to_dict()
