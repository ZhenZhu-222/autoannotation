const state = {
  jobId: "",
  preferCurrentLabels: false,
  page: 1,
  pageSize: 24,
  total: 0,
  lastSummary: {},
  classNames: {},
};

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function queryString() {
  return new URLSearchParams(window.location.search);
}

function fmtNum(value, digits = 3) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function displayPipelineName(pipeline) {
  const value = String(pipeline || "");
  if (value === "qwen_zero_shot") return "AI 零样本";
  return value || "model";
}

function normalizeClassNames(rawNames) {
  if (Array.isArray(rawNames)) {
    return Object.fromEntries(rawNames.map((name, idx) => [String(idx), String(name || "").trim()]).filter(([, name]) => name));
  }
  if (rawNames && typeof rawNames === "object") {
    return Object.fromEntries(Object.entries(rawNames).map(([id, name]) => [String(id), String(name || "").trim()]).filter(([, name]) => name));
  }
  return {};
}

function formatLabelText(rawText) {
  const text = String(rawText || "").trim();
  if (!text || text === "空txt" || text.startsWith("读取失败")) return text || "空txt";
  return text.split("\n").map((line) => {
    const parts = line.trim().split(/\s+/);
    if (!parts.length) return line;
    const className = state.classNames[String(parts[0])] || "";
    return className ? `${parts[0]}(${className}) ${parts.slice(1).join(" ")}` : line;
  }).join("\n");
}

async function api(url) {
  const resp = await fetch(url, { cache: "no-store" });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || `请求失败: ${resp.status}`);
  }
  return data;
}

async function fetchText(url) {
  if (!url) return "空txt";
  const raw = String(url || "").trim();
  const sep = raw.includes("?") ? "&" : "?";
  const resp = await fetch(`${raw}${sep}_ts=${Date.now()}`, { cache: "no-store" });
  if (!resp.ok) return `读取失败 (${resp.status})`;
  const text = (await resp.text()).trim();
  return text || "空txt";
}

async function renderDownloads() {
  try {
    const data = await api(`/api/annotate/jobs/${state.jobId}/artifacts`);
    const artifacts = data.artifacts || {};
    const links = [
      artifacts.zip_file ? `<a href="${artifacts.zip_file}" target="_blank">下载完整结果 ZIP</a>` : "",
      artifacts.cvat_yolo11_zip ? `<a href="${artifacts.cvat_yolo11_zip}" target="_blank">下载 CVAT YOLO 1.1 ZIP</a>` : "",
      artifacts.cvat_ultralytics_zip ? `<a href="${artifacts.cvat_ultralytics_zip}" target="_blank">下载 CVAT Ultralytics ZIP</a>` : "",
    ].filter(Boolean);
    $("downloads").innerHTML = links.join("");
  } catch (e) {
    $("downloads").innerHTML = `<span class="muted-text">${escapeHtml(e.message)}</span>`;
  }
}

function renderSummary(summary) {
  state.lastSummary = summary;
  const qualityPart = summary.pipeline === "qwen_zero_shot"
    ? `，精度 ${escapeHtml(summary.precision_mode || "fast")}（采样 ${summary.sample_count ?? 1}）` +
      `，低质拒绝 ${summary.quality_rejected_images ?? 0}` +
      `，平均质量 ${fmtNum(summary.avg_quality_score ?? 0, 3)}`
    : "";
  const pipelineName = displayPipelineName(summary.pipeline);
  $("summary").innerHTML =
    `任务摘要：操作人 ${escapeHtml(summary.operator || "anonymous")}，` +
    `轮次 ${escapeHtml(summary.round_tag || "未命名")}，模式 ${escapeHtml(summary.label_mode || "-")}，` +
    `流水线 ${escapeHtml(pipelineName)}${qualityPart}<br>` +
    `图片 ${summary.total_images ?? 0}，旧框 ${summary.total_existing_boxes ?? 0}，` +
    `新框 ${summary.total_new_boxes ?? 0}，最终框 ${summary.total_final_boxes ?? 0}，` +
    `尺寸校验失败 ${summary.size_check_failed_images ?? 0}`;
}

async function renderGrid(items) {
  const labels = await Promise.all(items.map((item) => fetchText(item.label_url)));
  const cards = items.map((item, idx) => {
    const sourceCell = item.source_url
      ? `<img src="${item.source_url}" alt="source">`
      : `<div class="preview-empty">源图缺失</div>`;
    const overlayCell = item.overlay_url
      ? `<img src="${item.overlay_url}" alt="overlay">`
      : `<div class="preview-empty">overlay缺失</div>`;
    return `
      <article class="preview-card">
        <div class="preview-head">
          <strong>${escapeHtml(item.filename)}</strong>
          <span>状态: ${escapeHtml(item.status || "-")}</span>
          <span>框: 旧${item.existing_boxes} / 新${item.new_boxes} / 最终${item.final_boxes}</span>
          <span>尺寸校验: ${Number(item.size_check_passed || 0) ? "通过" : "未通过"}</span>
          <span>质量: ${fmtNum(item.quality_score ?? 0, 3)} / 一致性: ${fmtNum(item.consensus_rate ?? 0, 3)}</span>
        </div>
        <div class="preview-cols">
          <div>
            <div class="preview-label">原图</div>
            ${sourceCell}
          </div>
          <div>
            <div class="preview-label">Overlay</div>
            ${overlayCell}
          </div>
          <div>
            <div class="preview-label">标签TXT（含类别名）</div>
            <pre class="preview-txt">${escapeHtml(formatLabelText(labels[idx]))}</pre>
          </div>
        </div>
        ${item.reject_reason ? `<div class="preview-error">拒绝原因: ${escapeHtml(item.reject_reason)}</div>` : ""}
        ${item.error ? `<div class="preview-error">错误: ${escapeHtml(item.error)}</div>` : ""}
        ${item.qwen_note ? `<div class="preview-error">AI备注: ${escapeHtml(item.qwen_note)}</div>` : ""}
      </article>
    `;
  });
  $("previewGrid").innerHTML = cards.join("") || `<div class="status-box">当前筛选下无数据</div>`;
}

async function loadPage() {
  const q = new URLSearchParams();
  q.set("page", String(state.page));
  q.set("page_size", String(state.pageSize));
  q.set("only_with_boxes", $("onlyWithBoxes").checked ? "true" : "false");
  q.set("keyword", $("keyword").value.trim());
  if (state.preferCurrentLabels) q.set("prefer_current_labels", "true");

  const data = await api(`/api/annotate/jobs/${state.jobId}/preview?${q.toString()}`);
  state.total = Number(data.total || 0);
  state.classNames = normalizeClassNames(data.class_names || data.summary?.class_names || {});
  const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
  if (state.page > totalPages) {
    state.page = totalPages;
    return loadPage();
  }

  $("previewSubTitle").textContent = `任务 ${state.jobId}`;
  renderSummary(data.summary || {});
  await renderGrid(data.items || []);

  $("pagerText").textContent = `第 ${state.page} / ${totalPages} 页，共 ${state.total} 条`;
  $("btnPrev").disabled = state.page <= 1;
  $("btnNext").disabled = state.page >= totalPages;
}

function bindEvents() {
  $("btnSearch").addEventListener("click", async () => {
    state.page = 1;
    await loadPage().catch((e) => {
      $("previewGrid").innerHTML = `<div class="status-box">${escapeHtml(e.message)}</div>`;
    });
  });
  $("btnPrev").addEventListener("click", async () => {
    if (state.page <= 1) return;
    state.page -= 1;
    await loadPage().catch(() => {});
  });
  $("btnNext").addEventListener("click", async () => {
    state.page += 1;
    await loadPage().catch(() => {});
  });
}

async function bootstrap() {
  state.jobId = queryString().get("job_id") || "";
  state.preferCurrentLabels = ["1", "true", "yes"].includes((queryString().get("prefer_current_labels") || "").toLowerCase());
  if (!state.jobId) {
    $("previewGrid").innerHTML = `<div class="status-box">缺少 job_id 参数</div>`;
    return;
  }
  bindEvents();
  await renderDownloads();
  await loadPage().catch((e) => {
    $("previewGrid").innerHTML = `<div class="status-box">${escapeHtml(e.message)}</div>`;
  });
}

bootstrap();
