let _session = null;

function renderLicense(status) {
  $("fingerprintInput").value = status.machine_fingerprint || "";
  const title = $("licenseTitle");
  const meta = $("licenseMeta");
  const mode = status.mode || (status.locked ? "locked" : "trial");
  if (title) title.textContent = mode === "licensed" ? "永久授权已激活" : "授权状态";
  if (mode === "licensed") {
    meta.textContent = `客户 ${status.customer || "-"}，license_id ${status.license_id || "-"}，签发日期 ${status.issued_at || "-"}。永久使用。`;
    return;
  }
  if (mode === "trial") {
    meta.textContent = `当前为 14 天试用，还剩 ${status.trial_days_left} 天。`;
    return;
  }
  meta.textContent = `${mode === "expired" ? "试用已到期" : "系统已锁定"}：${status.error || "请上传有效授权文件"}`;
}

async function loadStatus() {
  const resp = await fetch("/api/system/session", { credentials: "same-origin" });
  const data = await resp.json().catch(() => ({}));
  _session = data;
  if (!data.authenticated) {
    location.href = "/front/auth.html";
    return;
  }
  renderLicense(data.license || {});
  const canInstall = !!data.capabilities?.can_install_license;
  $("installPanel")?.classList.toggle("hidden", !canInstall);
}

async function installLicense() {
  const status = $("status");
  const file = $("licenseFile")?.files?.[0];
  if (!file) {
    status.textContent = "请选择授权文件";
    status.className = "hint error";
    return;
  }
  const form = new FormData();
  form.append("file", file);
  status.textContent = "上传并校验中...";
  status.className = "hint";
  const resp = await fetch("/api/system/license/install", {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.ok === false) {
    status.textContent = data.detail || "授权文件安装失败";
    status.className = "hint error";
    return;
  }
  status.textContent = "授权已安装";
  status.className = "hint";
  renderLicense(data.license || {});
}

document.getElementById("btnInstallLicense")?.addEventListener("click", () => {
  installLicense().catch((e) => {
    const status = $("status");
    status.textContent = e.message || "授权文件安装失败";
    status.className = "hint error";
  });
});

document.getElementById("btnCopyFingerprint")?.addEventListener("click", () => {
  const input = $("fingerprintInput");
  input?.select();
  navigator.clipboard?.writeText(input?.value || "").catch(() => {});
});

document.getElementById("btnBack")?.addEventListener("click", () => {
  if (_session?.license?.locked) location.href = "/front/auth.html";
  else location.href = "/";
});

loadStatus().catch((e) => {
  const status = $("status");
  status.textContent = e.message || "状态加载失败";
  status.className = "hint error";
});
