/* ========== Annotation ========== */
async function startAnnotate() {
  const modelId = $("modelSelect").value;
  const imagesetId = $("annotateImageset").value;
  const sourceMode = $("annotateSourceMode")?.value || "imageset";
  if (!modelId) { toast("请先选择模型"); return; }
  if (!imagesetId) { toast("请先选择图片集"); return; }
  const currentMappingSig = computeCurrentMappingSignature();
  if (!state.mappingConfirmed || state.mappingSignature !== currentMappingSig) {
    setMappingConfirmed(false, "映射未确认或已变更：请点击\"确认映射（必做）\"后再启动。");
    toast("请先确认类别映射");
    return;
  }
  const req = {
    model_id: modelId,
    imageset_id: imagesetId,
    selected_class_ids: getSelectedClassIds(),
    target_classes: [...state.targetClasses],
    conf: Number($("annotateConf").value || "0.25"),
    iou: Number($("annotateIou").value || "0.45"),
    device: $("annotateDevice").value.trim(),
    save_overlays: true,
    label_task: normalizeLabelTask($("annotateLabelTask")?.value || state.modelMeta[modelId]?.task),
    label_mode: $("labelMode").value,
    update_imageset_labels: $("updateImagesetLabels").checked,
    round_tag: $("roundTag").value.trim(),
    operator: $("annotateOperator").value.trim() || "anonymous",
    class_id_overrides: getClassIdOverrides(),
    mapping_confirmed: true,
  };
  $("annotateStatus").textContent = "创建打标任务...";
  if ($("artifacts")) $("artifacts").innerHTML = "";
  try {
    const job = await api("/api/annotate/jobs", { method: "POST", body: req });
    state.currentAnnotateJob = job.id;
    pollAnnotateJob(job.id);
  } catch (e) {
    $("annotateStatus").textContent = e.message;
  }
}

async function pollAnnotateJob(jobId) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/annotate/jobs/${jobId}`);
      $("annotateStatus").textContent = `打标 [${job.status}] ${(job.progress * 100).toFixed(1)}% ${job.progress_text || ""}`;
      if (job.status === "succeeded") {
        clearInterval(timer);
        toast("打标完成", "success");
        state.currentAnnotateJob = null;
        setMappingConfirmed(false, "打标已完成。如需下一轮，请重新确认映射。");
        const details = await api(`/api/annotate/jobs/${jobId}/artifacts`);
        renderArtifacts(details.artifacts, details.summary);
        await refreshImagesetSelects();
        loadInlinePreview($("annotateImageset")?.value, "annotateInlinePreview");
        if (state.sourceClasses.length) await renderClassChecklist();
        await loadGallery();
        await refreshHistoryPanel();
      }
      if (job.status === "failed") {
        clearInterval(timer);
        state.currentAnnotateJob = null;
        setMappingConfirmed(false, "打标失败。修正后请重新确认映射。");
        $("annotateStatus").textContent = `失败: ${job.error}`;
      }
      if (job.status === "cancelled") {
        clearInterval(timer);
        state.currentAnnotateJob = null;
        setMappingConfirmed(false, "打标已取消。如需重试请重新确认映射。");
        $("annotateStatus").textContent = `已取消: ${job.error || ""}`;
      }
    } catch (e) {
      clearInterval(timer);
      $("annotateStatus").textContent = e.message;
    }
  }, 1200);
}

function renderArtifacts(artifacts, summary, targetId = "artifacts") {
  const box = $(targetId);
  if (!box) return;
  if (targetId === "artifacts") {
    state.lastAnnotateRollbackApi = artifacts.rollback_api || "";
  } else if (targetId === "qwenArtifacts") {
    state.lastQwenRollbackApi = artifacts.rollback_api || "";
  }
  const fallbackJobId = state.currentAnnotateJob || state.currentQwenJob || "";
  const previewUrl = artifacts.preview_page_url || `/front/preview.html?job_id=${fallbackJobId}`;
  const overrides = summary.class_id_overrides || {};
  const overrideText = Object.keys(overrides).length
    ? Object.entries(overrides).map(([src, dst]) => `${src}\u2192${dst}`).join(", ")
    : "无";
  const sizeGuardText = summary.pipeline === "qwen_zero_shot"
    ? `, 尺寸校验失败 ${summary.size_check_failed_images || 0}` +
      `, 严格 ${summary.strict_size_check ? "开" : "关"}` +
      `, 精度 ${summary.precision_mode || "fast"}` +
      `, 采样 ${summary.sample_count || 1}` +
      `, 低质拒绝 ${summary.quality_rejected_images || 0}` +
      `, SVG失败 ${summary.svg_parse_failed_images || 0}`
    : "";
  const targetClassText = Array.isArray(summary.target_classes) && summary.target_classes.length
    ? `, 目标类别 ${summary.target_classes.length}` : "";
  const cvatLinks = [
    isAdminUser() && artifacts.cvat_yolo11_zip ? `<a href="${artifacts.cvat_yolo11_zip}" target="_blank">CVAT YOLO 1.1</a>` : "",
    isAdminUser() && artifacts.cvat_ultralytics_zip ? `<a href="${artifacts.cvat_ultralytics_zip}" target="_blank">CVAT Ultralytics</a>` : "",
    artifacts.label_id_map ? `<a href="${artifacts.label_id_map}" target="_blank">label_id_map.json</a>` : "",
  ].filter(Boolean).join("");
  const cleanupAction = isAdminUser() && artifacts.cleanup_api
    ? `<button class="btn-cleanup danger" data-api="${artifacts.cleanup_api}" data-target="${targetId}">删除产物</button>` : "";
  const resultZip = isAdminUser() && artifacts.zip_file
    ? `<a href="${artifacts.zip_file}" target="_blank">结果ZIP</a>`
    : "";
  box.innerHTML = `
    <div class="status-box">
      操作人 ${summary.operator || "anonymous"} | 轮次 ${summary.round_tag || "-"} | 模式 ${summary.label_mode || "-"} | 回写 ${summary.update_imageset_labels ? "是" : "否"} | 映射 ${overrideText}${targetClassText}${sizeGuardText}<br>
      图片 ${summary.total_images}，旧框 ${summary.total_existing_boxes ?? 0}，新框 ${summary.total_new_boxes ?? 0}，最终框 ${summary.total_final_boxes ?? summary.total_boxes}
    </div>
    <div class="actions">${cleanupAction}</div>
    <a href="${previewUrl}" target="_blank">对照画廊</a>
    ${resultZip}
    <a href="${artifacts.manifest_csv}" target="_blank">manifest.csv</a>
    <a href="${artifacts.run_meta_json}" target="_blank">run_meta.json</a>
    <a href="${artifacts.labels_dir}" target="_blank">labels</a>
    <a href="${artifacts.overlays_dir}" target="_blank">overlays</a>
    <a href="${artifacts.images_dir}" target="_blank">images</a>
    ${cvatLinks}
  `;
  const cleanupBtn = box.querySelector(".btn-cleanup");
  if (cleanupBtn) {
    cleanupBtn.addEventListener("click", () => cleanupRunArtifacts(cleanupBtn.dataset.api, cleanupBtn.dataset.target));
  }
}

async function rollbackRunLabels(apiPath, targetId = "") {
  if (!apiPath) return;
  if (!window.confirm("确认撤回本次打标对图片集 labels 的改动吗？")) return;
  try {
    const data = await api(apiPath, { method: "POST" });
    const msg = `已回滚：恢复 ${data.restored_files || 0}，删除 ${data.removed_files || 0}`;
    if (targetId === "qwenArtifacts") {
      $("qwenArtifacts").innerHTML = `<div class="status-box">${msg}</div>`;
    } else {
      if ($("artifacts")) $("artifacts").innerHTML = `<div class="status-box">${msg}</div>`;
    }
    toast(msg);
    await Promise.all([refreshImagesetSelects(), loadGallery(), refreshHistoryPanel()]);
  } catch (e) {
    toast(`回滚失败: ${e.message}`, "error");
  }
}

async function cleanupRunArtifacts(apiPath, targetId = "") {
  if (!apiPath) return;
  if (!window.confirm("确认删除本任务产物？")) return;
  try {
    const data = await api(apiPath, { method: "DELETE" });
    const msg = `已清理：目录 ${data.removed_dirs || 0}，文件 ${data.removed_files || 0}`;
    if (targetId === "qwenArtifacts") {
      $("qwenStatus").textContent = msg;
      $("qwenArtifacts").innerHTML = `<div class="status-box">${msg}</div>`;
    } else if (targetId === "artifacts") {
      $("annotateStatus").textContent = msg;
      $("artifacts").innerHTML = `<div class="status-box">${msg}</div>`;
    }
    toast(msg);
    await refreshHistoryPanel();
  } catch (e) {
    toast(`清理失败: ${e.message}`, "error");
  }
}

async function rollbackCurrentAnnotate() {
  const apiPath = state.lastAnnotateRollbackApi || (state.currentAnnotateJob ? `/api/annotate/jobs/${state.currentAnnotateJob}/rollback` : "");
  if (!apiPath) { toast("没有可撤回的权重打标任务"); return; }
  await rollbackRunLabels(apiPath, "artifacts");
}

async function cancelCurrentAnnotate() { await cancelJobById(state.currentAnnotateJob, "权重打标", "annotateStatus"); }
