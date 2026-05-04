async function login() {
  const username = document.getElementById("username")?.value?.trim() || "";
  const password = document.getElementById("password")?.value || "";
  const status = document.getElementById("status");
  if (!username || !password) {
    status.textContent = "请输入账户名和密码";
    status.className = "hint error";
    return;
  }
  status.textContent = "登录中...";
  status.className = "hint";
  const resp = await fetch("/api/system/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.ok === false) {
    status.textContent = data.detail || "登录失败";
    status.className = "hint error";
    return;
  }
  if (data.license?.locked) {
    location.href = "/front/license.html";
    return;
  }
  location.href = "/";
}

document.getElementById("btnLogin")?.addEventListener("click", () => {
  login().catch((e) => {
    const status = document.getElementById("status");
    status.textContent = e.message || "登录失败";
    status.className = "hint error";
  });
});

document.getElementById("password")?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    login().catch((e) => {
      const status = document.getElementById("status");
      status.textContent = e.message || "登录失败";
      status.className = "hint error";
    });
  }
});
