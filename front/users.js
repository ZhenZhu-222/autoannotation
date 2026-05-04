async function ensureAdmin() {
  const session = await api("/api/system/session");
  if (!session.authenticated) {
    location.href = "/front/auth.html";
    return false;
  }
  if (!session.capabilities?.can_manage_users) {
    location.href = "/";
    return false;
  }
  return true;
}

function fmtUserTime(iso) {
  if (!iso) return "-";
  try { return new Date(iso).toLocaleString("zh-CN"); } catch (_) { return iso; }
}

async function loadUsers() {
  const data = await api("/api/system/users");
  const box = $("usersTable");
  const rows = [
    '<div class="job-row head"><div>用户名</div><div>角色</div><div>状态</div><div>创建时间</div><div>最后登录</div><div>操作</div></div>',
  ];
  data.items.forEach((u) => {
    const enabledText = u.enabled ? "启用" : "禁用";
    rows.push(
      `<div class="job-row">` +
        `<div>${_esc(u.username)}</div>` +
        `<div>${_esc(u.role)}</div>` +
        `<div>${enabledText}</div>` +
        `<div>${fmtUserTime(u.created_at)}</div>` +
        `<div>${fmtUserTime(u.last_login_at)}</div>` +
        `<div>` +
          `<a href="#" class="user-toggle" data-id="${u.id}" data-enabled="${u.enabled ? "0" : "1"}">${u.enabled ? "禁用" : "启用"}</a> ` +
          `<a href="#" class="user-role" data-id="${u.id}" data-role="${u.role === "admin" ? "operator" : "admin"}">改为${u.role === "admin" ? "operator" : "admin"}</a> ` +
          `<a href="#" class="user-reset" data-id="${u.id}">重置密码</a> ` +
          `<a href="#" class="user-delete" data-id="${u.id}">删除</a>` +
        `</div>` +
      `</div>`
    );
  });
  box.innerHTML = rows.join("");
  box.querySelectorAll(".user-toggle").forEach((el) => {
    el.addEventListener("click", async (ev) => {
      ev.preventDefault();
      await api(`/api/system/users/${el.dataset.id}`, { method: "PATCH", body: { enabled: el.dataset.enabled === "1" } });
      await loadUsers();
    });
  });
  box.querySelectorAll(".user-role").forEach((el) => {
    el.addEventListener("click", async (ev) => {
      ev.preventDefault();
      await api(`/api/system/users/${el.dataset.id}`, { method: "PATCH", body: { role: el.dataset.role } });
      await loadUsers();
    });
  });
  box.querySelectorAll(".user-reset").forEach((el) => {
    el.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const pwd = window.prompt("输入新密码（至少 6 位）");
      if (!pwd) return;
      await api(`/api/system/users/${el.dataset.id}/password`, { method: "POST", body: { new_password: pwd } });
      toast("密码已重置", "success");
    });
  });
  box.querySelectorAll(".user-delete").forEach((el) => {
    el.addEventListener("click", async (ev) => {
      ev.preventDefault();
      if (!window.confirm("确认删除该用户？该操作会软删除并禁用账号。")) return;
      await api(`/api/system/users/${el.dataset.id}`, { method: "DELETE" });
      await loadUsers();
    });
  });
}

async function createUser() {
  const status = $("userStatus");
  try {
    await api("/api/system/users", {
      method: "POST",
      body: {
        username: $("newUsername")?.value.trim() || "",
        password: $("newPassword")?.value || "",
        role: $("newRole")?.value || "operator",
      },
    });
    if (status) status.textContent = "用户已创建";
    if ($("newUsername")) $("newUsername").value = "";
    if ($("newPassword")) $("newPassword").value = "";
    await loadUsers();
  } catch (e) {
    if (status) status.textContent = e.message;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  if (!(await ensureAdmin())) return;
  on("btnRefreshUsers", "click", loadUsers);
  on("btnCreateUser", "click", createUser);
  await loadUsers();
});
