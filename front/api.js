async function api(url, options = {}) {
  const opts = { ...options };
  if (opts.body && !(opts.body instanceof FormData) && typeof opts.body !== "string") {
    opts.headers = { ...(opts.headers || {}), "Content-Type": "application/json" };
    opts.body = JSON.stringify(opts.body);
  }
  let resp;
  try {
    resp = await fetch(url, opts);
  } catch (error) {
    const reason = String(error?.message || "").trim();
    throw new Error(reason ? `接口不可达：${reason}` : "接口不可达，请确认服务是否正在运行");
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    let message = `请求失败: ${resp.status}`;
    if (typeof data.detail === "string" && data.detail.trim()) {
      message = data.detail.trim();
    } else if (Array.isArray(data.detail) && data.detail.length) {
      message = data.detail.map((item) => item?.msg || item?.type || JSON.stringify(item)).join("；");
    }
    throw new Error(message);
  }
  return data;
}

function _uploadFormatBytes(bytes) {
  if (typeof formatBytes === "function") return formatBytes(bytes);
  const size = Number(bytes || 0);
  if (!Number.isFinite(size) || size <= 0) return "0B";
  if (size < 1024) return `${size}B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)}KB`;
  return `${(size / (1024 * 1024)).toFixed(1)}MB`;
}

function _uploadEsc(value) {
  if (typeof _esc === "function") return _esc(value);
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderUploadProgress(target, state = {}) {
  const el = typeof target === "string" ? $(target) : target;
  if (!el) return;
  const loaded = Number(state.loaded || 0);
  const total = Number(state.total || 0);
  const hasTotal = Number.isFinite(total) && total > 0;
  const percent = hasTotal ? Math.max(0, Math.min(100, Math.round((loaded / total) * 100))) : 0;
  const width = hasTotal ? percent : 36;
  const label = state.label || "上传";
  const phase = state.phase || (hasTotal ? `${_uploadFormatBytes(loaded)} / ${_uploadFormatBytes(total)}` : `${_uploadFormatBytes(loaded)} 已上传`);
  const percentText = hasTotal ? `${percent}%` : "上传中";
  el.innerHTML = `
    <div class="upload-progress">
      <div class="upload-progress-head">
        <span>${_uploadEsc(label)}</span>
        <strong>${_uploadEsc(percentText)}</strong>
      </div>
      <div class="upload-progress-track">
        <div class="upload-progress-fill${hasTotal ? "" : " indeterminate"}" style="width:${width}%"></div>
      </div>
      <div class="upload-progress-meta">${_uploadEsc(phase)}</div>
    </div>
  `;
}

async function uploadFormWithProgress(url, options = {}) {
  const {
    method = "POST",
    body,
    statusEl = null,
    label = "上传",
    processingText = "上传完成，服务端处理中...",
    onProgress = null,
    onUploadComplete = null,
  } = options;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url, true);
    xhr.upload.onprogress = (event) => {
      const progress = {
        label,
        loaded: event.loaded,
        total: event.lengthComputable ? event.total : 0,
      };
      if (typeof onProgress === "function") onProgress(progress, event);
      else renderUploadProgress(statusEl, progress);
    };
    xhr.upload.onload = () => {
      const progress = { label, loaded: 1, total: 1, phase: processingText };
      if (typeof onUploadComplete === "function") onUploadComplete(progress);
      else renderUploadProgress(statusEl, progress);
    };
    xhr.onerror = () => reject(new Error("接口不可达：服务未启动、端口不对或连接被中断；目录上传已改为逐文件上传，若仍失败请检查服务日志、网络和磁盘空间"));
    xhr.onabort = () => reject(new Error("上传已取消"));
    xhr.onload = () => {
      let data = {};
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch (_) {
        data = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data);
        return;
      }
      let message = `请求失败: ${xhr.status}`;
      if (typeof data.detail === "string" && data.detail.trim()) {
        message = data.detail.trim();
      } else if (Array.isArray(data.detail) && data.detail.length) {
        message = data.detail.map((item) => item?.msg || item?.type || JSON.stringify(item)).join("；");
      }
      reject(new Error(message));
    };
    renderUploadProgress(statusEl, { label, loaded: 0, total: 1, phase: "准备上传..." });
    xhr.send(body);
  });
}

function withNoCache(url) {
  const raw = String(url || "").trim();
  if (!raw) return "";
  const sep = raw.includes("?") ? "&" : "?";
  return `${raw}${sep}_ts=${Date.now()}`;
}

async function fetchTextNoCache(url) {
  if (!url) return "";
  const resp = await fetch(withNoCache(url), { cache: "no-store" });
  return resp.ok ? resp.text() : "";
}
