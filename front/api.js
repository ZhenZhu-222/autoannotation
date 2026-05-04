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
