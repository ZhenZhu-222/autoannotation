function syncQwenPrecisionModeUi() {
  const mode = $("qwenPrecisionMode")?.value || "strict";
  ["qwenSampleCount", "qwenMaxCalibrationErrorPx", "qwenMinConsensusRate", "qwenDuplicateIou"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = mode !== "strict";
  });
}

async function rollbackCurrentQwen() {
  const apiPath = state.lastQwenRollbackApi || (state.currentQwenJob ? `/api/annotate/jobs/${state.currentQwenJob}/rollback` : "");
  if (!apiPath) { toast("没有可撤回的 AI 打标任务"); return; }
  await rollbackRunLabels(apiPath, "qwenArtifacts");
}

async function cancelCurrentQwen() { await cancelJobById(state.currentQwenJob, "AI 打标", "qwenStatus"); }

/* ========== Qwen Annotate ========== */
async function pollQwenAnnotateJob(jobId) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/qwen/annotate/jobs/${jobId}`);
      $("qwenStatus").textContent = `AI 打标 [${job.status}] ${(job.progress * 100).toFixed(1)}% ${job.progress_text || ""}`;
      if (job.status === "succeeded") {
        clearInterval(timer);
        toast("AI 打标完成", "success");
        state.currentQwenJob = null;
        $("qwenStatus").textContent = "AI 打标完成。可开始下一轮。";
        const details = await api(`/api/annotate/jobs/${jobId}/artifacts`);
        renderArtifacts(details.artifacts, details.summary, "qwenArtifacts");
        await refreshImagesetSelects();
        loadInlinePreview($("aiImagesetSelect")?.value, "aiInlinePreview");
        await loadGallery();
        await refreshHistoryPanel();
      }
      if (job.status === "failed") {
        clearInterval(timer);
        state.currentQwenJob = null;
        $("qwenStatus").textContent = `失败: ${job.error}`;
      }
      if (job.status === "cancelled") {
        clearInterval(timer);
        state.currentQwenJob = null;
        $("qwenStatus").textContent = `已取消: ${job.error || ""}`;
      }
    } catch (e) {
      clearInterval(timer);
      $("qwenStatus").textContent = e.message;
    }
  }, 1200);
}

async function qwenAutoAnnotate() {
  const imagesetId = $("aiImagesetSelect")?.value;
  if (!imagesetId) { toast("请先选择目标图片集"); return; }
  const desc = $("qwenPrompt")?.value.trim() || "";
  const refs = Array.from($("qwenRefImages")?.files || []);
  if (!desc && !refs.length) { toast("请至少输入文字描述，或上传参考图片"); return; }
  const form = new FormData();
  form.append("imageset_id", imagesetId);
  form.append("description", desc);
  form.append("qwen_model", $("qwenModelName")?.value.trim() || "");
  form.append("api_key", $("qwenApiKey")?.value.trim() || "");
  form.append("label_task", normalizeLabelTask($("qwenLabelTask")?.value || "detect"));
  form.append("label_mode", $("aiLabelMode")?.value || "append");
  form.append("update_imageset_labels", $("aiUpdateImagesetLabels")?.checked ? "true" : "false");
  form.append("round_tag", $("aiRoundTag")?.value.trim() || "");
  form.append("operator", $("aiAnnotateOperator")?.value.trim() || "anonymous");
  form.append("save_overlays", "true");
  form.append("strict_size_check", $("qwenStrictSizeCheck")?.checked ? "true" : "false");
  form.append("size_retry", $("qwenSizeRetry")?.value || "1");
  form.append("precision_mode", $("qwenPrecisionMode")?.value || "strict");
  form.append("sample_count", $("qwenSampleCount")?.value || "3");
  form.append("min_consensus_rate", $("qwenMinConsensusRate")?.value || "0.67");
  form.append("duplicate_iou", $("qwenDuplicateIou")?.value || "0.98");
  refs.forEach((f) => form.append("files", f));
  $("qwenStatus").textContent = "创建 AI 打标任务...";
  if ($("qwenArtifacts")) $("qwenArtifacts").innerHTML = "";
  try {
    const job = await uploadFormWithProgress("/api/qwen/annotate/jobs", {
      method: "POST",
      body: form,
      statusEl: "qwenStatus",
      label: refs.length ? `上传参考图 ${refs.length} 张` : "创建 AI 打标任务",
      processingText: "上传完成，正在创建 AI 打标任务...",
    });
    state.currentQwenJob = job.id;
    const refInput = $("qwenRefImages");
    if (refInput) refInput.value = "";
    syncQwenRefFilesUi();
    pollQwenAnnotateJob(job.id);
  } catch (e) {
    $("qwenStatus").textContent = e.message;
    toast(`AI 任务创建失败: ${e.message}`, "error");
  }
}
