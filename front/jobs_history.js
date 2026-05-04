async function batchDeleteHistory() {
  const checked = Array.from(document.querySelectorAll(".history-check:checked"));
  if (!checked.length) { toast("请先勾选要删除的记录"); return; }
  let ok = 0, fail = 0;
  for (const cb of checked) {
    try {
      await api(`/api/jobs/history/${cb.dataset.jobId}?with_artifacts=true`, { method: "DELETE" });
      ok++;
    } catch (e) {
      fail++;
    }
  }
  toast(`已删除 ${ok} 条${fail ? `，失败 ${fail} 条` : ""}`, ok ? "success" : "error");
  await Promise.all([refreshHistoryPanel(), refreshImagesetSelects()]);
}

function toggleSelectAllHistory() {
  const boxes = document.querySelectorAll(".history-check");
  const allChecked = Array.from(boxes).every(cb => cb.checked);
  boxes.forEach(cb => { cb.checked = !allChecked; });
}

async function deleteHistoryEntry(jobId) {
  if (!jobId) return;
  try {
    await api(`/api/jobs/history/${jobId}?with_artifacts=true&with_imageset=true`, { method: "DELETE" });
    toast("已删除", "success");
    await Promise.all([refreshHistoryPanel(), refreshImagesetSelects()]);
  } catch (e) {
    toast(`删除失败: ${e.message}`, "error");
  }
}

async function openImagesetFromHistory(imagesetId) {
  if (!imagesetId) { toast("该记录未关联图片集"); return; }
  await refreshImagesetSelects();
  const target = $("galleryImageset");
  if (!target) return;
  target.value = imagesetId;
  setActivePanel("panel-video");
  window.scrollTo({ top: 0, behavior: "smooth" });
  await loadGallery().catch(() => {});
}

async function cancelJobById(jobId, label, statusTargetId) {
  if (!jobId) { toast(`没有可取消的${label}任务`); return; }
  if (!window.confirm(`确认取消${label}任务吗？`)) return;
  try {
    const data = await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    const msg = `${label}: ${data.status}${data.cancel_requested ? "（已请求取消）" : ""}`;
    if ($(statusTargetId)) $(statusTargetId).textContent = msg;
    toast(msg);
  } catch (e) {
    toast(`取消失败: ${e.message}`, "error");
  }
}

/* ========== Jobs & History Panels ========== */
async function refreshJobsPanel() {
  try {
    const data = await api("/api/jobs");
    const box = $("jobsTable");
    const count = $("jobsCount");
    if (count) count.textContent = String(data.items.length);
    const rows = [
      '<div class="job-row head"><div>Job ID</div><div>类型</div><div>状态</div><div>进度</div><div>信息</div><div>操作</div></div>',
    ];
    data.items.slice(0, 20).forEach((job) => {
      const canCancel = job.status === "queued" || job.status === "running";
      const op = canCancel ? `<a href="#" class="job-cancel" data-job-id="${job.id}">取消</a>` : "-";
      rows.push(
        `<div class="job-row">` +
          `<div>${job.id}</div>` +
          `<div>${displayJobType(job.job_type)}</div>` +
          `<div><span class="status-badge ${job.status}">${job.status}</span></div>` +
          `<div>${(job.progress * 100).toFixed(1)}%</div>` +
          `<div>${job.error || job.progress_text || ""}</div>` +
          `<div>${op}</div>` +
        `</div>`
      );
    });
    box.innerHTML = rows.join("");
    box.querySelectorAll(".job-cancel").forEach((el) => {
      el.addEventListener("click", async (ev) => {
        ev.preventDefault();
        await cancelJobById(el.dataset.jobId || "", "任务", "");
        await refreshJobsPanel();
      });
    });
  } catch (_) {}
}

async function refreshHistoryPanel() {
  try {
    const q = new URLSearchParams();
    q.set("limit", "200");
    if ($("historyJobType")?.value) q.set("job_type", $("historyJobType").value);
    if ($("historyOperator")?.value?.trim()) q.set("operator", $("historyOperator").value.trim());
    const data = await api(`/api/jobs/history?${q.toString()}`);
    const box = $("historyTable");
    const count = $("historyCount");
    if (count) count.textContent = String(data.items.length);
    const fmtTime = (iso) => {
      if (!iso) return "-";
      try { const d = new Date(iso); return d.toLocaleString("zh-CN", { month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit", second:"2-digit" }); } catch { return iso; }
    };
    const rows = [
      '<div class="job-row head"><div></div><div>创建时间</div><div>完成时间</div><div>任务</div><div>状态</div><div>操作人</div><div>关键信息</div><div>操作</div></div>',
    ];
    data.items.forEach((item) => {
      const summary = item.result_summary || {};
      const artifacts = summary.artifacts || {};
      const infoParts = [];
      if (summary.round_tag) infoParts.push(`round=${summary.round_tag}`);
      if (summary.imageset_id) infoParts.push(`set=${summary.imageset_id}`);
      if (summary.total_final_boxes !== undefined) infoParts.push(`boxes=${summary.total_final_boxes}`);
      if (summary.saved_images !== undefined) infoParts.push(`frames=${summary.saved_images}`);
      if (item.error) infoParts.push(`err=${item.error}`);
      const effectLinks = [];
      const isAnnotateLike = item.job_type === "annotate" || item.job_type === "qwen_annotate";
      if (isAnnotateLike) {
        const previewUrl = artifacts.preview_page_url || `/front/preview.html?job_id=${item.job_id}`;
        effectLinks.push(`<a href="${previewUrl}" target="_blank">画廊</a>`);
        if (isAdminUser() && artifacts.zip_file) effectLinks.push(`<a href="${artifacts.zip_file}" target="_blank">ZIP</a>`);
        if (isAdminUser() && artifacts.cvat_ultralytics_zip) effectLinks.push(`<a href="${artifacts.cvat_ultralytics_zip}" target="_blank">CVAT</a>`);
        if (artifacts.can_rollback && artifacts.rollback_api) effectLinks.push(`<a href="#" class="history-rollback" data-api="${artifacts.rollback_api}">撤回</a>`);
        if (isAdminUser() && artifacts.cleanup_api) effectLinks.push(`<a href="#" class="history-cleanup" data-api="${artifacts.cleanup_api}">清理</a>`);
      } else if (item.job_type === "extract" && summary.imageset_id) {
        effectLinks.push(`<a href="#" class="history-open-imageset" data-imageset-id="${summary.imageset_id}">打开图片集</a>`);
      }
      if (isAdminUser()) effectLinks.push(`<a href="#" class="history-delete" data-job-id="${item.job_id}">删除</a>`);
      rows.push(
        `<div class="job-row">` +
          `<div><input type="checkbox" class="history-check" data-job-id="${item.job_id}"></div>` +
          `<div>${fmtTime(item.created_at)}</div>` +
          `<div>${fmtTime(item.finished_at)}</div>` +
          `<div>${displayJobType(item.job_type)}<br><small>${item.job_id}</small></div>` +
          `<div><span class="status-badge ${item.status}">${item.status}</span></div>` +
          `<div>${item.operator || "anonymous"}</div>` +
          `<div>${infoParts.join(" ; ")}</div>` +
          `<div>${effectLinks.join(" ") || "-"}</div>` +
        `</div>`
      );
    });
    box.innerHTML = rows.join("");
    box.querySelectorAll(".history-rollback").forEach((el) => {
      el.addEventListener("click", async (ev) => { ev.preventDefault(); await rollbackRunLabels(el.dataset.api || "", "history"); });
    });
    box.querySelectorAll(".history-cleanup").forEach((el) => {
      el.addEventListener("click", async (ev) => { ev.preventDefault(); await cleanupRunArtifacts(el.dataset.api || "", "history"); });
    });
    box.querySelectorAll(".history-delete").forEach((el) => {
      el.addEventListener("click", async (ev) => { ev.preventDefault(); await deleteHistoryEntry(el.dataset.jobId || ""); });
    });
    box.querySelectorAll(".history-open-imageset").forEach((el) => {
      el.addEventListener("click", async (ev) => { ev.preventDefault(); await openImagesetFromHistory(el.dataset.imagesetId || ""); });
    });
  } catch (_) {}
}
