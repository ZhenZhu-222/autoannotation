function $(id) {
  return document.getElementById(id);
}

function _esc(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function toast(msg, variant) {
  const el = $("toast");
  if (!el) return;
  const tone = variant || "info";
  const icons = {
    success: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    error: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    warning: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  };
  el.className = `toast toast-${tone}`;
  el.innerHTML = `<span class="toast-icon">${icons[tone] || icons.info}</span><span>${msg}</span>`;
  setTimeout(() => { el.className = "toast hidden"; }, 2800);
}

function on(id, event, handler) {
  const el = $(id);
  if (!el) return;
  el.addEventListener(event, handler);
}

function formatBytes(bytes) {
  const size = Number(bytes || 0);
  if (!Number.isFinite(size) || size <= 0) return "0B";
  if (size < 1024) return `${size}B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)}KB`;
  return `${(size / (1024 * 1024)).toFixed(1)}MB`;
}

function normalizeLabelTask(task) {
  const text = String(task || "").toLowerCase();
  return ["detect", "segment", "obb"].includes(text) ? text : "detect";
}

function labelTaskLabel(task) {
  const normalized = normalizeLabelTask(task);
  if (normalized === "segment") return "seg";
  if (normalized === "obb") return "obb";
  return "detect";
}

function displayJobType(jobType) {
  const value = String(jobType || "");
  if (value === "qwen_annotate") return "AI 打标";
  return value || "-";
}
