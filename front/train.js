/* ========== Module 5: YOLO Training ========== */

function updateTrainDatasetInfo() {
  const id = $("trainImageset")?.value;
  const box = $("trainDatasetInfo");
  if (!box) return;
  if (!id) { box.textContent = "请选择数据集"; return; }
  const meta = state.imagesetMeta[id];
  if (!meta) { box.textContent = "未找到数据集信息"; return; }
  const labelCount = meta.label_count || 0;
  box.textContent = labelCount > 0
    ? `${meta.name} — ${labelTaskLabel(meta.label_task)}，图:${meta.image_count} 标:${labelCount}，可训练`
    : `${meta.name} — 图:${meta.image_count} 标:0，请先打标`;
}

function syncTrainBaseModelUI() {
  const sel = $("trainBaseModelSelect");
  const wrap = $("trainSystemModelWrap");
  if (!sel || !wrap) return;
  wrap.style.display = sel.value === "__system__" ? "" : "none";
}

function syncTrainTaskUi(forceBase = false) {
  const task = normalizeLabelTask($("trainTask")?.value || "detect");
  const sel = $("trainBaseModelSelect");
  if (!sel || sel.value === "__system__") return;
  const defaults = { detect: "yolo11n.pt", segment: "yolo11n-seg.pt", obb: "yolo11n-obb.pt" };
  const currentTask = sel.value.includes("obb") ? "obb" : (sel.value.includes("seg") ? "segment" : "detect");
  if (forceBase || currentTask !== task) {
    sel.value = defaults[task] || "yolo11n.pt";
  }
}

function resetTrainArtifacts() {
  const artifacts = $("trainArtifacts");
  if (artifacts) artifacts.innerHTML = "";
}

async function startTrain() {
  const imagesetId = $("trainImageset")?.value;
  if (!imagesetId) { toast("请先选择图片集", "warning"); return; }
  const meta = state.imagesetMeta[imagesetId];
  if (meta && !meta.label_count) { toast("该数据集无标注，请先打标", "warning"); return; }

  const baseSel = $("trainBaseModelSelect")?.value || "yolo11n.pt";
  let baseModel = baseSel;
  let baseModelId = "";
  if (baseSel === "__system__") {
    baseModelId = $("trainSystemModelSelect")?.value || "";
    if (!baseModelId) { toast("请选择系统模型", "warning"); return; }
    baseModel = "";
  }

  const body = {
    imageset_id: imagesetId,
    epochs: parseInt($("trainEpochs")?.value || "50"),
    batch_size: parseInt($("trainBatchSize")?.value || "16"),
    img_size: parseInt($("trainImgSize")?.value || "640"),
    base_model: baseModel,
    base_model_id: baseModelId,
    task: normalizeLabelTask($("trainTask")?.value || meta?.label_task || "detect"),
    device: ($("trainDevice")?.value || "").trim(),
    save_to_system: !!$("trainSaveToSystem")?.checked,
    operator: ($("trainOperator")?.value || "anonymous").trim() || "anonymous",
  };

  try {
    resetTrainArtifacts();
    const data = await api("/api/train/jobs", { method: "POST", body });
    state.currentTrainJob = data.job_id;
    $("trainStatus").textContent = `训练任务已提交: ${data.job_id}`;
    toast("训练任务已提交", "success");
    pollTrainJob(data.job_id);
  } catch (e) {
    const message = e?.message || "训练任务创建失败";
    if (message.includes("YOLO 训练要求类 ID 连续")) {
      $("trainStatus").textContent = message;
      toast(message, "warning");
      return;
    }
    $("trainStatus").textContent = message;
    toast(message, "error");
  }
}

async function pollTrainJob(jobId) {
  const box = $("trainStatus");
  const poll = async () => {
    try {
      const data = await api(`/api/train/jobs/${jobId}`);
      const status = data.status;
      const pct = Math.round((data.progress || 0) * 100);
      const text = data.progress_text || "";
      box.textContent = `[${status}] ${pct}% ${text}`;

      if (status === "succeeded") {
        toast("训练完成", "success");
        const result = data.result || {};
        renderTrainArtifacts(result);
        refreshModels();
        return;
      }
      if (status === "failed") {
        toast(`训练失败: ${data.error || "未知"}`, "error");
        return;
      }
      if (status === "cancelled") {
        toast("训练已取消", "warning");
        return;
      }
      setTimeout(poll, 3000);
    } catch (e) {
      box.textContent = `轮询失败: ${e.message}`;
    }
  };
  poll();
}

function renderTrainArtifacts(result) {
  const el = $("trainArtifacts");
  if (!el) return;
  const arts = result.artifacts || {};
  const lines = [];
  lines.push(`<div class="status-box">训练完成 — 类别数: ${result.class_count || 0}, Epochs: ${result.epochs || 0}</div>`);
  if (result.saved_model_id) {
    lines.push(`<div class="help-text">已保存到系统模型库 (ID: ${result.saved_model_id})</div>`);
  }
  if (isAdminUser() && arts.best_pt) {
    lines.push(`<a href="${arts.best_pt}" download class="btn-link minor" style="margin-top:6px;display:inline-block">下载 best.pt</a>`);
  }
  const previewItems = Array.isArray(result.prediction_preview) ? result.prediction_preview : [];
  if (previewItems.length) {
    lines.push(
      `<details class="ip-details" style="margin-top:10px">
        <summary class="ip-summary">训练后模型预测预览（样本） （点击展开）</summary>
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-top:10px">` +
      previewItems.map((item) => `
        <div class="ip-card">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div>
              <div class="preview-label">原图</div>
              <img src="${withNoCache(item.source_url)}" alt="${_esc(item.filename)} 原图">
            </div>
            <div>
              <div class="preview-label">训练后预测</div>
              <img src="${withNoCache(item.overlay_url)}" alt="${_esc(item.filename)} 预测">
            </div>
          </div>
          <div class="ip-meta">
            <span class="ip-name" title="${_esc(item.filename)}">${_esc(item.filename)}</span>
            <span class="ip-label-badge ${item.prediction_count ? "has" : "none"}">${item.prediction_count ? `预测 ${item.prediction_count} 个` : "无预测"}</span>
            <pre class="ip-txt">${_esc(item.prediction_text || "无预测")}</pre>
          </div>
        </div>
      `).join("") +
      `</div>
      </details>`,
    );
  } else {
    lines.push(`<div class="help-text" style="margin-top:10px">当前未生成训练后预测预览，可先下载模型或重新训练后再看。</div>`);
  }
  el.innerHTML = lines.join("");
}

async function cancelCurrentTrain() {
  if (!state.currentTrainJob) { toast("无训练任务", "warning"); return; }
  try {
    await api(`/api/jobs/${state.currentTrainJob}/cancel`, { method: "POST" });
    toast("取消请求已发送", "info");
  } catch (e) {
    toast(e.message, "error");
  }
}

async function downloadModel() {
  return downloadModelFromSelect("downloadModelSelect");
}

async function downloadModelFromSelect(selectId) {
  const sel = $(selectId);
  const modelId = sel?.value;
  if (!modelId) { toast("请选择模型", "warning"); return; }
  try {
    const url = `/api/models/${modelId}/download`;
    const resp = await fetch(url);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const cd = resp.headers.get("content-disposition") || "";
    const filenameStar = cd.match(/(?:^|;)\s*filename\*=UTF-8''([^;]+)/i);
    const match = cd.match(/(?:^|;)\s*filename="?([^";]+)"?/i);
    const meta = state.modelMeta[modelId] || {};
    const headerName = filenameStar
      ? decodeURIComponent(filenameStar[1])
      : (match ? decodeURIComponent(match[1]) : "");
    const selectedName = meta.name || sel?.options?.[sel.selectedIndex]?.textContent?.split(" [")[0] || "";
    a.download = ensureModelFilenameExtension(headerName || selectedName || "model", meta.model_type);
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    toast("下载开始", "success");
  } catch (e) {
    toast(e.message, "error");
  }
}
