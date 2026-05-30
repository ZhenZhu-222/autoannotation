const state = {
  currentExtractJob: null,
  currentAnnotateJob: null,
  currentQwenJob: null,
  currentTrainJob: null,
  currentUser: null,
  capabilities: {},
  license: null,
  lastAnnotateRollbackApi: "",
  lastQwenRollbackApi: "",
  currentGalleryImages: [],
  galleryPage: 1,
  galleryPageSize: 120,
  galleryTotal: 0,
  imagesetMeta: {},
  imagesetClassNames: {},
  modelMeta: {},
  refineImages: [],
  refineImageData: null,
  refineCurrentImageId: "",
  refineSelectedBoxId: "",
  refineDefaultClassValue: "",
  refineDirty: false,
  refinePendingClassNames: [],
  refineCreateMode: false,
  refineDraftPoints: [],
  refineDraftHoverPoint: null,
  refineInteraction: null,
  sourceClasses: [],
  targetClasses: [],
  classPresets: [],
  mappingConfirmed: false,
  mappingSignature: "",
};

const BOTTOM_PANEL_KEY = "autoannotation.bottomPanelCollapsed";
const DEFAULT_MAX_VIDEO_UPLOAD_BYTES = 2000 * 1024 * 1024;
const MODEL_DOWNLOAD_EXTENSIONS = [".pt", ".onnx"];

/* ========== Top Tab Navigation ========== */
function setActivePanel(panelId) {
  document.querySelectorAll(".switch-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });
  document.querySelectorAll(".nav-tab[data-panel-target]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.panelTarget === panelId);
  });
  if (panelId === "panel-refine" && state.refineImageData) {
    requestAnimationFrame(() => {
      renderRefineOverlay();
      _refineFocusWorkspace();
    });
  }
}

function bindSectionNav() {
  document.querySelectorAll(".nav-tab[data-panel-target]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.panelTarget;
      if (target) setActivePanel(target);
    });
  });
}

/* ========== Bottom Panel ========== */
function setBottomPanelCollapsed(collapsed) {
  const panel = $("bottomPanel");
  const btn = $("btnToggleBottomPanel");
  if (!panel) return;
  panel.classList.toggle("collapsed", collapsed);
  if (btn) {
    btn.innerHTML = collapsed
      ? '<span class="icon"><svg viewBox="0 0 24 24"><polyline points="18 15 12 9 6 15"/></svg></span> 展开'
      : '<span class="icon"><svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg></span> 收起';
  }
  try { localStorage.setItem(BOTTOM_PANEL_KEY, collapsed ? "1" : "0"); } catch (_) {}
}

function toggleBottomPanel() {
  const panel = $("bottomPanel");
  setBottomPanelCollapsed(!panel?.classList.contains("collapsed"));
}

function initBottomPanel() {
  let collapsed = false;
  try { collapsed = localStorage.getItem(BOTTOM_PANEL_KEY) === "1"; } catch (_) {}
  setBottomPanelCollapsed(collapsed);

  document.querySelectorAll(".bottom-tab[data-bottom-target]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.bottomTarget;
      if (!target) return;
      document.querySelectorAll(".bottom-tab").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".bottom-section").forEach((s) => s.classList.toggle("active", s.id === target));
      // auto-expand if collapsed
      const panel = $("bottomPanel");
      if (panel?.classList.contains("collapsed")) {
        setBottomPanelCollapsed(false);
      }
    });
  });
}

/* ========== Helpers ========== */
function isTypingTarget(target) {
  if (!target) return false;
  const tag = String(target.tagName || "").toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function setSelectValue(id, value) {
  const el = $(id);
  if (el && value) el.value = value;
}


async function openNativeFilePicker(inputId) {
  const input = $(inputId);
  if (!input) return;
  try {
    if (typeof input.showPicker === "function") {
      input.showPicker();
      return;
    }
    input.click();
  } catch (_) {
    toast("当前浏览器未允许调起文件选择，请先点原生文件输入框再试");
  }
}

function validateVideoFile(file) {
  if (!file) return { ok: false, message: "请选择视频文件" };
  const size = Number(file.size || 0);
  if (size > DEFAULT_MAX_VIDEO_UPLOAD_BYTES) {
    return {
      ok: false,
      message: `视频过大，当前限制 ${formatBytes(DEFAULT_MAX_VIDEO_UPLOAD_BYTES)}，请先压缩或裁剪`,
    };
  }
  return { ok: true, message: "" };
}

function syncVideoFileUi() {
  const input = $("videoFile");
  const name = $("videoFileName");
  if (!input || !name) return;
  const file = input.files && input.files[0];
  if (!file) {
    name.textContent = "未选择任何文件";
    return;
  }
  const validation = validateVideoFile(file);
  name.textContent = validation.ok
    ? `${file.name} · ${formatBytes(file.size)}`
    : `${file.name} · ${formatBytes(file.size)} · 超出限制`;
}

function syncResourceFolderUi() {
  const input = $("resourceFolderInput");
  const name = $("resourceFolderFileName");
  if (!input || !name) return;
  const files = Array.from(input.files || []);
  if (!files.length) {
    name.textContent = "未选择目录";
    return;
  }
  const root = String(files[0].webkitRelativePath || files[0].name).split("/")[0];
  name.textContent = `${root || "已选目录"} · ${files.length} 个文件`;
}

function syncModelUploadUi() {
  const model = $("modelFile")?.files?.[0];
  const classes = $("classesFile")?.files?.[0];
  if ($("modelFileName")) {
    $("modelFileName").textContent = model ? `${model.name} · ${formatBytes(model.size)}` : "未选择权重文件";
  }
  if ($("classesFileName")) {
    $("classesFileName").textContent = classes ? `${classes.name} · ${formatBytes(classes.size)}` : "未选择类别文件";
  }
}

function syncQwenRefFilesUi() {
  const input = $("qwenRefImages");
  const summary = $("qwenRefSummary");
  const list = $("qwenRefList");
  const name = $("qwenRefFilesName");
  if (!input || !summary || !list) return;
  const files = Array.from(input.files || []);
  if (!files.length) {
    if (name) name.textContent = "未选择任何文件";
    summary.textContent = "未添加参考图片";
    list.innerHTML = "";
    return;
  }
  const totalBytes = files.reduce((acc, f) => acc + (Number(f.size) || 0), 0);
  if (name) {
    name.textContent = files.length === 1
      ? `${files[0].name} · ${formatBytes(files[0].size)}`
      : `已选择 ${files.length} 个文件`;
  }
  summary.textContent = `已添加 ${files.length} 张参考图，总大小 ${formatBytes(totalBytes)}`;
  list.innerHTML = files
    .slice(0, 12)
    .map((f) => `<span class="file-pill">${f.name} · ${formatBytes(f.size)}</span>`)
    .join("");
}

async function checkHealth() {
  try {
    const h = await api("/health");
    $("healthBadge").textContent = `服务正常`;
    $("healthBadge").style.color = "var(--ok)";
  } catch (e) {
    $("healthBadge").textContent = `异常: ${e.message}`;
    $("healthBadge").style.color = "var(--danger)";
  }
}

async function loadSessionContext() {
  // 启动时只从 /api/system/session 读取当前用户和能力位。
  // 后续按钮显隐都基于 capabilities，不在前端重复写角色规则。
  const data = await api("/api/system/session");
  state.currentUser = data.user || null;
  state.capabilities = data.capabilities || {};
  state.license = data.license || null;
  const username = $("currentUsername");
  const role = $("currentRoleBadge");
  if (username) username.textContent = state.currentUser?.username || data.username || "未登录";
  if (role) {
    const roleText = state.currentUser?.role || "";
    role.textContent = roleText || "-";
    role.classList.toggle("admin", roleText === "admin");
  }
  applyCapabilities();
}

function isAdminUser() {
  return !!state.capabilities?.can_manage_users || state.currentUser?.role === "admin";
}

function applyCapabilities() {
  // 前端隐藏只是体验层；真正安全判断仍以后端权限校验为准。
  const isAdmin = isAdminUser();
  document.querySelectorAll("[data-admin-only]").forEach((el) => {
    el.classList.toggle("hidden", !isAdmin);
  });
  $("taskMetrics")?.classList.toggle("operator-view", !isAdmin);
  [
    "btnDeleteImageset", "btnDownloadImageset", "btnBatchDelete", "btnDownloadAnnotateImageset",
    "btnDownloadAiImageset", "btnDownloadRemapImageset", "btnDownloadRefineImageset",
    "btnDownloadTrainImageset", "btnDownloadSelectedModel", "btnDeleteModel",
    "btnDownloadModel", "btnDownloadTrainSystemModel", "btnSelectAllHistory", "btnBatchDeleteHistory",
    "btnAssignImageset",
  ].forEach((id) => {
    const el = $(id);
    if (el) el.classList.toggle("hidden", !isAdmin);
  });
  renderTaskBoard();
  renderResourceImagesets();
  renderResourceModels();
}

function toggleUserMenu() {
  $("userMenuDropdown")?.classList.toggle("hidden");
}

async function logout() {
  await fetch("/api/system/logout", { method: "POST", credentials: "same-origin" }).catch(() => {});
  location.href = "/front/auth.html";
}

function openPasswordModal() {
  ["oldPasswordInput", "newPasswordInput", "confirmPasswordInput"].forEach((id) => { if ($(id)) $(id).value = ""; });
  if ($("passwordStatus")) $("passwordStatus").textContent = "";
  $("passwordModal")?.classList.remove("hidden");
  $("userMenuDropdown")?.classList.add("hidden");
}

function closePasswordModal() {
  $("passwordModal")?.classList.add("hidden");
}

async function submitPasswordChange() {
  const oldPassword = $("oldPasswordInput")?.value || "";
  const newPassword = $("newPasswordInput")?.value || "";
  const confirm = $("confirmPasswordInput")?.value || "";
  const status = $("passwordStatus");
  if (newPassword.length < 6) {
    if (status) status.textContent = "新密码至少 6 位";
    return;
  }
  if (newPassword !== confirm) {
    if (status) status.textContent = "两次新密码不一致";
    return;
  }
  try {
    await api("/api/system/users/me/password", {
      method: "PATCH",
      body: { old_password: oldPassword, new_password: newPassword },
    });
    closePasswordModal();
    toast("密码已修改", "success");
  } catch (e) {
    if (status) status.textContent = e.message;
  }
}

async function assignCurrentImageset() {
  await openAssignModal($("galleryImageset")?.value || "");
}

function closeAssignModal() {
  $("assignModal")?.classList.add("hidden");
}

function getAssignSelectedIds() {
  return Array.from(document.querySelectorAll(".assign-operator-check:checked")).map((el) => el.value);
}

function updateAssignSummary(namesOverride = null) {
  const checks = Array.from(document.querySelectorAll(".assign-operator-check:checked"));
  const names = namesOverride || checks.map((el) => el.dataset.username || el.value);
  const count = $("assignSelectedCount");
  const summary = $("assignSummary");
  if (count) count.textContent = `已选 ${names.length} 人`;
  if (summary) {
    summary.textContent = names.length
      ? `将发布给：${names.join("、")}`
      : "当前未选择业务人员；保存后这个图片集不会分派给任何 operator。";
  }
}

function renderAssignImagesetInfo() {
  const imagesetId = $("assignImagesetSelect")?.value || "";
  const info = $("assignImagesetInfo");
  const meta = state.imagesetMeta?.[imagesetId];
  if (!info) return;
  if (!meta) {
    info.textContent = "未选择图片集";
    return;
  }
  const assigned = meta.assignee_usernames?.length ? meta.assignee_usernames.join("、") : "未分派";
  info.textContent = `${meta.name} · 图片 ${meta.image_count} · 当前分派：${assigned}`;
}

async function openAssignModal(preferredImagesetId = "") {
  try {
    if (!isAdminUser()) return;
    await refreshImagesetSelects();
    const sel = $("assignImagesetSelect");
    if (!sel) return;
    sel.innerHTML = "";
    Object.values(state.imagesetMeta || {}).forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.imageset_id;
      opt.textContent = `${it.name}（图:${it.image_count}）`;
      sel.appendChild(opt);
    });
    if (preferredImagesetId) sel.value = preferredImagesetId;
    if (!sel.value && sel.options.length) sel.value = sel.options[0].value;
    await renderAssignOperators();
    renderAssignImagesetInfo();
    $("assignStatus").textContent = "";
    $("assignModal")?.classList.remove("hidden");
    $("userMenuDropdown")?.classList.add("hidden");
  } catch (e) {
    toast(e.message || "打开分派失败", "error");
  }
}

async function renderAssignOperators() {
  const imagesetId = $("assignImagesetSelect")?.value || "";
  const box = $("assignOperatorList");
  if (!box) return;
  if (!imagesetId) {
    box.innerHTML = '<div class="muted-text">暂无图片集</div>';
    return;
  }
  const users = await api("/api/system/users");
  const operators = (users.items || []).filter((u) => u.role === "operator" && u.enabled && !u.deleted);
  const assigned = new Set(state.imagesetMeta[imagesetId]?.assignee_ids || []);
  if (!operators.length) {
    box.innerHTML = '<div class="assign-empty">暂无可分派的业务人员，请先到“业务人员”页面创建 operator。</div>';
    updateAssignSummary([]);
    renderAssignImagesetInfo();
    return;
  }
  box.innerHTML = operators.map((u) => `
    <label class="assign-person">
      <input type="checkbox" class="assign-operator-check" value="${_esc(u.id)}" data-username="${_esc(u.username)}" ${assigned.has(u.id) ? "checked" : ""}>
      <span class="assign-person-main">
        <span class="assign-person-name">${_esc(u.username)}</span>
        <span class="assign-person-meta">operator · ${_esc(u.id)}</span>
      </span>
    </label>
  `).join("");
  box.querySelectorAll(".assign-operator-check").forEach((input) => {
    input.addEventListener("change", () => updateAssignSummary());
  });
  updateAssignSummary();
  renderAssignImagesetInfo();
}

async function submitAssignModal() {
  try {
    const imagesetId = $("assignImagesetSelect")?.value || "";
    if (!imagesetId) { toast("请先选择图片集", "warning"); return; }
    const ids = getAssignSelectedIds();
    const submit = $("btnSubmitAssign");
    const status = $("assignStatus");
    if (submit) submit.disabled = true;
    if (status) status.textContent = "正在保存分派...";
    const saved = await api(`/api/imagesets/${imagesetId}/assignees`, { method: "POST", body: { user_ids: ids } });
    const savedIds = new Set((saved.assignees || []).map((u) => u.id));
    const missing = ids.filter((id) => !savedIds.has(id));
    if (missing.length) {
      throw new Error(`后端未确认这些用户的分派：${missing.join(", ")}`);
    }
    await refreshImagesetSelects();
    await renderAssignOperators();
    renderTaskBoard();
    const names = (saved.assignees || []).map((u) => u.username);
    updateAssignSummary(names);
    if (status) status.textContent = names.length ? `已保存：${names.join("、")} 可以看到该图片集。` : "已保存：该图片集当前没有分派给业务人员。";
    toast(names.length ? `已分派给 ${names.join("、")}` : "已清空分派", "success");
  } catch (e) {
    if ($("assignStatus")) $("assignStatus").textContent = e.message || "保存失败";
    toast(e.message || "保存分派失败", "error");
  } finally {
    const submit = $("btnSubmitAssign");
    if (submit) submit.disabled = false;
  }
}

async function assignCurrentImagesetLegacyPrompt() {
  try {
    if (!isAdminUser()) return;
    const imagesetId = $("galleryImageset")?.value;
    if (!imagesetId) { toast("请先选择图片集", "warning"); return; }
    const users = await api("/api/system/users");
    const operators = (users.items || []).filter((u) => u.role === "operator" && u.enabled && !u.deleted);
    const current = state.imagesetMeta[imagesetId]?.assignee_usernames || [];
    const names = operators.map((u) => u.username).join(", ");
    const input = window.prompt(`输入要分派的 operator 用户名，多个用逗号分隔。\n可选：${names}`, current.join(", "));
    if (input === null) return;
    const requested = input.split(",").map((x) => x.trim()).filter(Boolean);
    const ids = requested.map((name) => {
      const found = operators.find((u) => u.username === name);
      if (!found) throw new Error(`operator 不存在或不可用: ${name}`);
      return found.id;
    });
    await api(`/api/imagesets/${imagesetId}/assignees`, { method: "POST", body: { user_ids: ids } });
    toast("分派已保存", "success");
    await refreshImagesetSelects();
  } catch (e) {
    toast(e.message || "分派失败", "error");
  }
}


/* ========== Data Refresh ========== */
async function refreshImagesetSelects() {
  const data = await api("/api/imagesets");
  state.imagesetMeta = {};
  data.items.forEach((it) => { state.imagesetMeta[it.imageset_id] = it; });
  const selects = [$("galleryImageset"), $("annotateImageset"), $("aiImagesetSelect"), $("remapImageset"), $("trainImageset"), $("refineImageset")].filter(Boolean);
  selects.forEach((sel) => {
    const old = sel.value;
    sel.innerHTML = "";
    data.items.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.imageset_id;
      const owner = it.creator_username ? ` · 创建:${it.creator_username}` : "";
      const assigned = isAdminUser() && it.assignee_usernames?.length ? ` · 分派:${it.assignee_usernames.join("/")}` : "";
      opt.textContent = `${it.name} [${labelTaskLabel(it.label_task)}] (图:${it.image_count} 标:${it.label_count || 0})${owner}${assigned}`;
      sel.appendChild(opt);
    });
    if (old) sel.value = old;
  });
  renderTaskBoard();
  renderResourceImagesets();
  refreshTaskLedger().catch(() => {});
}

function taskAssigneeText(item) {
  const names = item.assignee_usernames || [];
  return names.length ? names.join("、") : "未分派";
}

function assignmentStatusLabel(status) {
  const value = String(status || "assigned");
  if (value === "submitted") return "待验收";
  if (value === "rejected") return "已驳回";
  if (value === "accepted") return "已验收";
  if (value === "canceled") return "已取消";
  return "处理中";
}

function assignmentStatusClass(status) {
  const value = String(status || "assigned");
  if (value === "submitted") return "submitted";
  if (value === "rejected") return "rejected";
  if (value === "accepted") return "accepted";
  if (value === "canceled") return "canceled";
  return "assigned";
}

function taskOwnershipText(item) {
  if (isAdminUser()) return `创建者：${item.creator_username || "未知"}`;
  if (item.creator_id && state.currentUser?.user_id === item.creator_id) return "我创建的";
  if ((item.assignee_ids || []).includes(state.currentUser?.user_id)) return "分派给我";
  return "可操作";
}

function taskActionButton(action, label, item, extraClass = "minor") {
  return `<button class="${extraClass} task-action" type="button" data-action="${action}" data-imageset-id="${_esc(item.imageset_id)}">${label}</button>`;
}

function assignmentActionButton(action, label, item, userId, extraClass = "minor") {
  return `<button class="${extraClass} task-action" type="button" data-action="${action}" data-imageset-id="${_esc(item.imageset_id)}" data-assignee-id="${_esc(userId)}">${label}</button>`;
}

function latestAcceptedAssignment(item) {
  const history = Array.isArray(item.assignee_history) ? item.assignee_history : [];
  const accepted = history.filter((entry) => entry?.status === "accepted");
  if (!accepted.length) return null;
  return accepted.reduce((latest, entry) => {
    const latestTime = String(latest.reviewed_at || latest.submitted_at || latest.assigned_at || "");
    const entryTime = String(entry.reviewed_at || entry.submitted_at || entry.assigned_at || "");
    return entryTime.localeCompare(latestTime) > 0 ? entry : latest;
  }, accepted[0]);
}

function renderAcceptedAssignmentRow(assignment) {
  if (!assignment) return "";
  const reviewer = assignment.reviewer_username || "未知";
  const assignee = assignment.username || assignment.id || "未知";
  const reviewedAt = assignment.reviewed_at ? ` · ${formatTaskTime(assignment.reviewed_at)}` : "";
  const note = assignment.review_note ? ` · 备注：${assignment.review_note}` : "";
  return `
    <div class="task-assignee-row task-assignee-row-accepted">
      <span class="task-assignee-name">已验收</span>
      <span class="task-state accepted">验收人：${_esc(reviewer)}</span>
      <span class="task-review-info">对象：${_esc(assignee)}${_esc(reviewedAt)}${_esc(note)}</span>
    </div>
  `;
}

function formatTaskTime(value) {
  const raw = String(value || "");
  if (!raw) return "-";
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw.slice(0, 19);
  return d.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function ledgerStatusClass(status) {
  const value = String(status || "");
  if (["accepted", "succeeded", "saved"].includes(value)) return "accepted";
  if (["rejected", "failed"].includes(value)) return "rejected";
  if (["submitted", "running", "queued"].includes(value)) return "submitted";
  if (["canceled", "cancelled"].includes(value)) return "canceled";
  return assignmentStatusClass(value);
}

function renderTaskLedgerRow(item) {
  const actor = item.actor_username || "未知";
  const target = item.target_username && item.target_username !== actor ? ` → ${item.target_username}` : "";
  const imagesetName = item.imageset_name || item.imageset_id || "系统任务";
  const imagesetSub = item.imageset_id ? `数据集：${item.imageset_id}` : (item.job_id ? `Job：${item.job_id}` : "");
  const noteParts = [];
  if (item.detail) noteParts.push(item.detail);
  if (item.note) noteParts.push(`备注：${item.note}`);
  if (item.job_id && item.event_type === "job") noteParts.push(item.job_id);
  return `
    <div class="task-ledger-row">
      <div>${_esc(formatTaskTime(item.occurred_at))}</div>
      <div class="task-ledger-action">${_esc(item.action || "-")}</div>
      <div class="task-ledger-main">
        <strong>${_esc(imagesetName)}</strong>
        <span>${_esc(imagesetSub)}</span>
      </div>
      <div>
        <span class="task-state ${ledgerStatusClass(item.status)}">${_esc(item.status_label || item.status || "已记录")}</span>
      </div>
      <div class="task-ledger-note">
        <strong>${_esc(actor + target)}</strong>
        <div class="task-ledger-muted">${_esc(noteParts.join(" · ") || "-")}</div>
      </div>
    </div>
  `;
}

async function refreshTaskLedger() {
  const box = $("taskLedgerTable");
  if (!box) return;
  try {
    const data = await api("/api/jobs/ledger?limit=120");
    const items = data.items || [];
    if (!items.length) {
      box.innerHTML = '<div class="task-ledger-empty">还没有流水记录。分派、提交验收、验收、自动打标、抽帧和训练完成后会出现在这里。</div>';
      return;
    }
    const rows = [
      '<div class="task-ledger-row head"><div>时间</div><div>动作</div><div>对象</div><div>状态</div><div>人员 / 备注</div></div>',
      ...items.map(renderTaskLedgerRow),
    ];
    box.innerHTML = rows.join("");
  } catch (e) {
    box.innerHTML = `<div class="task-ledger-empty">流水加载失败：${_esc(e.message || "未知错误")}</div>`;
  }
}

async function clearTaskLedger() {
  if (!isAdminUser()) return;
  const ok = window.confirm("确认格式化任务流水单？\n\n只清空页面里的流水记录，不删除数据集、图片、模型、任务产物。");
  if (!ok) return;
  try {
    const data = await api("/api/jobs/ledger", { method: "DELETE" });
    await refreshTaskLedger();
    toast(`流水已格式化，隐藏 ${data.hidden_rows || 0} 条旧记录`, "success");
  } catch (e) {
    toast(`格式化失败：${e.message}`, "error");
  }
}

function renderTaskAssignmentRows(item) {
  const isAdmin = isAdminUser();
  const ids = item.assignee_ids || [];
  const names = item.assignee_usernames || [];
  const states = item.assignee_states || {};
  let visibleIds = ids;
  if (!isAdmin) {
    visibleIds = ids.filter((id) => id === state.currentUser?.user_id);
  }
  if (!visibleIds.length) {
    const accepted = latestAcceptedAssignment(item);
    if (accepted) return renderAcceptedAssignmentRow(accepted);
    if (!isAdmin && item.creator_id === state.currentUser?.user_id) {
      return '<div class="task-assignee-row"><span>我创建的图片集</span><span class="task-state assigned">自有业务</span></div>';
    }
    return '<div class="task-assignee-row"><span>暂无业务人员</span><span class="task-state canceled">未分派</span></div>';
  }
  return visibleIds.map((userId) => {
    const index = ids.indexOf(userId);
    const name = names[index] || userId;
    const assignment = states[userId] || {};
    const status = assignment.status || "assigned";
    const submittedText = status === "submitted" && assignment.submitted_at ? ` · ${formatTaskTime(assignment.submitted_at)}` : "";
    let actions = "";
    if (isAdmin && status === "submitted") {
      actions = `
        ${assignmentActionButton("approve-review", "验收通过", item, userId, "primary")}
        ${assignmentActionButton("reject-review", "驳回", item, userId, "minor")}
      `;
    } else if (!isAdmin && userId === state.currentUser?.user_id) {
      if (status === "submitted") {
        actions = '<span class="task-waiting">等待管理员验收</span>';
      } else {
        actions = assignmentActionButton("submit-review", status === "rejected" ? "修改后重新提交" : "提交验收", item, userId, "primary");
      }
    }
    return `
      <div class="task-assignee-row">
        <span class="task-assignee-name">${_esc(name)}</span>
        <span class="task-state ${assignmentStatusClass(status)}">${assignmentStatusLabel(status)}${_esc(submittedText)}</span>
        <span class="task-assignee-actions">${actions}</span>
      </div>
    `;
  }).join("");
}

function renderTaskBoard() {
  const box = $("taskBoard");
  if (!box) return;
  const items = Object.values(state.imagesetMeta || {})
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  const isAdmin = isAdminUser();
  const total = items.length;
  const assignedCount = items.filter((it) => (it.assignee_ids || []).length > 0).length;
  const unassignedCount = total - assignedCount;
  const submittedCount = items.filter((it) => Object.values(it.assignee_states || {}).some((st) => st?.status === "submitted")).length;
  if ($("taskMetricTotal")) $("taskMetricTotal").textContent = String(total);
  if ($("taskMetricAssigned")) $("taskMetricAssigned").textContent = isAdmin ? String(submittedCount) : String(assignedCount);
  if ($("taskMetricUnassigned")) $("taskMetricUnassigned").textContent = String(unassignedCount);
  if ($("taskMetricRole")) $("taskMetricRole").textContent = state.currentUser?.role || "-";
  const subtitle = $("taskBoardSubtitle");
  if (subtitle) {
    subtitle.textContent = isAdmin
      ? "管理人员在这里发布/分派数据集任务，并查看每个数据集当前分给了谁。"
      : "这里是你当前可处理的数据集。未被分派的数据集不会显示。";
  }
  if (!items.length) {
    box.innerHTML = `
      <div class="task-empty">
        <strong>${isAdmin ? "还没有图片集任务" : "当前没有分派给你的业务"}</strong>
        <span>${isAdmin ? "先上传视频抽帧或导入图片集，然后在这里分派给业务人员。" : "请联系管理员在任务中心把图片集分派给你。"}</span>
      </div>
    `;
    return;
  }
  box.innerHTML = items.map((item) => {
    const owner = taskOwnershipText(item);
    const adminButtons = isAdmin ? taskActionButton("assign", "分派人员", item, "primary") : "";
    return `
      <article class="task-card">
        <div class="task-card-main">
          <div class="task-title-row">
            <h3>${_esc(item.name)}</h3>
            <span class="task-chip">${_esc(labelTaskLabel(item.label_task))}</span>
          </div>
          <div class="task-meta">
            <span>${_esc(owner)}</span>
            <span>图片 ${Number(item.image_count || 0)}</span>
            <span>标注 ${Number(item.label_count || 0)}</span>
            <span>${_esc(formatTaskTime(item.created_at))}</span>
          </div>
          <div class="task-assignees">
            ${renderTaskAssignmentRows(item)}
          </div>
        </div>
        <div class="task-card-actions">
          ${adminButtons}
          ${taskActionButton("view", "查看图片", item)}
          ${taskActionButton("refine", "数据精修", item)}
          ${taskActionButton("annotate", "权重打标", item)}
          ${taskActionButton("qwen", "AI打标", item)}
          ${taskActionButton("train", "训练", item)}
        </div>
      </article>
    `;
  }).join("");
}

async function submitTaskReview(imagesetId, userId) {
  const note = window.prompt("提交验收备注（可留空）：", "");
  if (note === null) return;
  await api(`/api/imagesets/${imagesetId}/assignees/${userId}/submit`, {
    method: "POST",
    body: { submit_note: note },
  });
  await refreshImagesetSelects();
  toast("已提交验收，等待管理员处理", "success");
}

async function reviewTaskAssignment(imagesetId, userId, approved) {
  const note = approved ? "" : window.prompt("驳回原因（建议填写）：", "");
  if (note === null) return;
  if (approved && !window.confirm("确认验收通过？通过后该任务会从业务人员的当前任务列表移除。")) return;
  await api(`/api/imagesets/${imagesetId}/assignees/${userId}/review`, {
    method: "POST",
    body: { approved, review_note: note || "" },
  });
  await refreshImagesetSelects();
  toast(approved ? "验收通过，任务已归档" : "已驳回，任务保留给业务人员继续处理", approved ? "success" : "warning");
}

async function openTaskImageset(imagesetId, action, assigneeId = "") {
  if (!imagesetId) return;
  if (action === "assign") {
    await openAssignModal(imagesetId);
    return;
  }
  if (action === "submit-review") {
    await submitTaskReview(imagesetId, state.currentUser?.user_id || "");
    return;
  }
  if (action === "approve-review") {
    await reviewTaskAssignment(imagesetId, assigneeId, true);
    return;
  }
  if (action === "reject-review") {
    await reviewTaskAssignment(imagesetId, assigneeId, false);
    return;
  }
  if (action === "view") {
    setActivePanel("panel-video");
    setSelectValue("galleryImageset", imagesetId);
    state.galleryPage = 1;
    await loadGallery();
    return;
  }
  if (action === "refine") {
    setActivePanel("panel-refine");
    setSelectValue("refineImageset", imagesetId);
    await loadRefineImageset({ skipDirty: true });
    return;
  }
  if (action === "annotate") {
    setActivePanel("panel-annotate");
    setSelectValue("annotateImageset", imagesetId);
    loadInlinePreview(imagesetId, "annotateInlinePreview");
    return;
  }
  if (action === "qwen") {
    setActivePanel("panel-qwen");
    setSelectValue("aiImagesetSelect", imagesetId);
    loadInlinePreview(imagesetId, "aiInlinePreview");
    return;
  }
  if (action === "train") {
    setActivePanel("panel-train");
    setSelectValue("trainImageset", imagesetId);
    const meta = state.imagesetMeta[imagesetId];
    if (meta?.label_task && $("trainTask")) $("trainTask").value = normalizeLabelTask(meta.label_task);
    syncTrainTaskUi(true);
    updateTrainDatasetInfo();
    resetTrainArtifacts();
    loadInlinePreview(imagesetId, "trainInlinePreview", { collapsed: true });
  }
}


function modelExtension(modelType) {
  const ext = `.${String(modelType || "").trim().toLowerCase().replace(/^\.+/, "")}`;
  return MODEL_DOWNLOAD_EXTENSIONS.includes(ext) ? ext : "";
}

function ensureModelFilenameExtension(filename, modelType) {
  const ext = modelExtension(modelType);
  let name = String(filename || "").replace(/\\/g, "/").split("/").pop().trim() || "model";
  if (!ext) return name;
  const lower = name.toLowerCase();
  if (lower.endsWith(ext)) return name;
  const currentExt = MODEL_DOWNLOAD_EXTENSIONS.find((item) => lower.endsWith(item));
  if (currentExt) return `${name.slice(0, -currentExt.length)}${ext}`;
  return `${name}${ext}`;
}

async function refreshModels() {
  const data = await api("/api/models");
  state.modelMeta = {};
  (data.items || []).forEach((it) => { state.modelMeta[it.model_id] = it; });
  const selects = [$("modelSelect"), $("downloadModelSelect"), $("trainSystemModelSelect")].filter(Boolean);
  selects.forEach((sel) => {
    const old = sel.value;
    sel.innerHTML = "";
    data.items.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.model_id;
      const displayName = ensureModelFilenameExtension(it.name, it.model_type);
      const owner = it.creator_username ? ` · 创建:${it.creator_username}` : "";
      opt.textContent = `${displayName} [${it.model_type}/${labelTaskLabel(it.task)}] classes=${it.classes_count}${owner}`;
      sel.appendChild(opt);
    });
    if (old) sel.value = old;
  });
  renderResourceModels();
}

/* ========== Resource Center ==========
   资源中心只做资源维护：数据集导入/改名/删除/下载/分派，以及权重上传/改名/删除/下载。
   打标、精修、训练仍在各自业务面板里执行，避免把“资源管理”和“生产操作”混在一起。 */
function assignmentSummaryForResource(item) {
  const names = item.assignee_usernames || [];
  if (!names.length) return "未分派";
  return names.join("、");
}

function currentUserAssignmentStatus(item) {
  const userId = state.currentUser?.user_id || "";
  const assignment = item.assignee_states?.[userId] || {};
  return assignmentStatusLabel(assignment.status || "assigned");
}

function updateResourceDatasetMetrics(items) {
  if ($("resourceMetricDatasets")) $("resourceMetricDatasets").textContent = String(items.length);
  if ($("resourceMetricImages")) {
    const totalImages = items.reduce((sum, it) => sum + Number(it.image_count || 0), 0);
    $("resourceMetricImages").textContent = String(totalImages);
  }
}

function renderResourceImagesets() {
  const box = $("resourceImagesetList");
  const items = Object.values(state.imagesetMeta || {})
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  updateResourceDatasetMetrics(items);
  if (!box) return;
  const isAdmin = isAdminUser();
  if (!items.length) {
    box.innerHTML = '<div class="resource-empty">暂无数据集。请先在上方上传图片目录或视频抽帧。</div>';
    return;
  }
  box.innerHTML = items.map((it) => {
    const adminActions = isAdmin ? `
      <button class="minor resource-action" type="button" data-kind="imageset" data-action="assign" data-id="${_esc(it.imageset_id)}">分派</button>
      <button class="minor resource-action" type="button" data-kind="imageset" data-action="download" data-id="${_esc(it.imageset_id)}">下载</button>
      <button class="danger resource-action" type="button" data-kind="imageset" data-action="delete" data-id="${_esc(it.imageset_id)}">删除</button>
    ` : "";
    const statusText = isAdmin ? `分派：${assignmentSummaryForResource(it)}` : currentUserAssignmentStatus(it);
    return `
      <article class="resource-row">
        <div class="resource-row-main">
          <div class="resource-title-line">
            <strong title="${_esc(it.name)}">${_esc(it.name)}</strong>
            <span class="resource-chip">${_esc(labelTaskLabel(it.label_task))}</span>
            <span class="resource-chip neutral">${_esc(it.source || "imageset")}</span>
          </div>
          <div class="resource-meta">
            <span>ID ${_esc(it.imageset_id)}</span>
            <span>创建者 ${_esc(it.creator_username || "未知")}</span>
            <span>图片 ${Number(it.image_count || 0)}</span>
            <span>标注 ${Number(it.label_count || 0)}</span>
            <span>${_esc(statusText)}</span>
            <span>${_esc(formatTaskTime(it.created_at))}</span>
          </div>
        </div>
        <div class="resource-row-actions">
          <button class="primary resource-action" type="button" data-kind="imageset" data-action="view" data-id="${_esc(it.imageset_id)}">预览</button>
          <button class="minor resource-action" type="button" data-kind="imageset" data-action="rename" data-id="${_esc(it.imageset_id)}">改名</button>
          ${adminActions}
        </div>
      </article>
    `;
  }).join("");
}

function renderResourceModels() {
  const box = $("resourceModelList");
  const items = Object.values(state.modelMeta || {})
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  if ($("resourceMetricModels")) $("resourceMetricModels").textContent = String(items.length);
  if (!box) return;
  const isAdmin = isAdminUser();
  if (!items.length) {
    box.innerHTML = '<div class="resource-empty">暂无权重。请先上传 .pt/.onnx，或训练后保存到系统模型库。</div>';
    return;
  }
  box.innerHTML = items.map((it) => {
    const displayName = ensureModelFilenameExtension(it.name, it.model_type);
    const adminActions = isAdmin ? `
      <button class="minor resource-action" type="button" data-kind="model" data-action="download" data-id="${_esc(it.model_id)}">下载</button>
      <button class="danger resource-action" type="button" data-kind="model" data-action="delete" data-id="${_esc(it.model_id)}">删除</button>
    ` : "";
    return `
      <article class="resource-row">
        <div class="resource-row-main">
          <div class="resource-title-line">
            <strong title="${_esc(displayName)}">${_esc(displayName)}</strong>
            <span class="resource-chip">${_esc(it.model_type || "-")}</span>
            <span class="resource-chip neutral">${_esc(labelTaskLabel(it.task))}</span>
          </div>
          <div class="resource-meta">
            <span>ID ${_esc(it.model_id)}</span>
            <span>创建者 ${_esc(it.creator_username || "未知")}</span>
            <span>类别 ${Number(it.classes_count || 0)}</span>
            <span>${_esc(formatTaskTime(it.created_at))}</span>
          </div>
        </div>
        <div class="resource-row-actions">
          <button class="primary resource-action" type="button" data-kind="model" data-action="use" data-id="${_esc(it.model_id)}">用于打标</button>
          <button class="minor resource-action" type="button" data-kind="model" data-action="rename" data-id="${_esc(it.model_id)}">改名</button>
          ${adminActions}
        </div>
      </article>
    `;
  }).join("");
}

async function uploadResourceFolderAsImageset() {
  const files = Array.from($("resourceFolderInput")?.files || []);
  if (!files.length) {
    toast("请选择图片目录", "warning");
    return;
  }
  const browserRoot = String(files[0].webkitRelativePath || files[0].name).split("/")[0];
  const name = $("resourceFolderName")?.value.trim() || browserRoot || `folder_${Date.now()}`;

  if ($("resourceFolderStatus")) $("resourceFolderStatus").textContent = "目录上传中...";
  try {
    const data = await uploadFolderFilesWithSession(files, name, "resourceFolderStatus");
    if ($("resourceFolderStatus")) $("resourceFolderStatus").textContent = datasetUploadSummary(data);
    await refreshImagesetSelects();
    setSelectValue("galleryImageset", data.imageset_id);
    setSelectValue("annotateImageset", data.imageset_id);
    setSelectValue("aiImagesetSelect", data.imageset_id);
    state.galleryPage = 1;
    await loadGallery().catch(() => {});
    toast("数据集已导入资源中心", "success");
  } catch (e) {
    if ($("resourceFolderStatus")) $("resourceFolderStatus").textContent = e.message;
    toast(`目录上传失败: ${e.message}`, "error");
  }
}

async function openResourceImageset(imagesetId) {
  if (!imagesetId) return;
  setSelectValue("galleryImageset", imagesetId);
  state.galleryPage = 1;
  await loadGallery();
  $("galleryGrid")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function renameImagesetById(imagesetId) {
  const meta = state.imagesetMeta[imagesetId];
  if (!meta) { toast("数据集不存在", "warning"); return; }
  const newName = window.prompt("输入新名称:", meta.name || "");
  if (!newName || newName.trim() === meta.name) return;
  const form = new FormData();
  form.append("name", newName.trim());
  await api(`/api/imagesets/${imagesetId}/rename`, { method: "PATCH", body: form });
  toast("数据集已改名", "success");
  await refreshImagesetSelects();
}

async function deleteImagesetById(imagesetId) {
  const meta = state.imagesetMeta[imagesetId];
  const name = meta?.name || imagesetId;
  if (!window.confirm(`确定删除数据集「${name}」及其全部图片和标签吗？`)) return;
  await api(`/api/imagesets/${imagesetId}`, { method: "DELETE" });
  toast("数据集已删除", "success");
  await refreshImagesetSelects();
  if ($("galleryImageset")?.value === imagesetId) {
    $("galleryGrid").innerHTML = "";
    if ($("galleryStatus")) $("galleryStatus").textContent = "当前数据集已删除";
  }
}

function downloadImagesetById(imagesetId) {
  if (!imagesetId) { toast("请选择数据集", "warning"); return; }
  window.open(`/api/imagesets/${imagesetId}/download`, "_blank");
}

async function renameModelById(modelId) {
  const meta = state.modelMeta[modelId];
  if (!meta) { toast("模型不存在", "warning"); return; }
  const currentName = meta.name || "";
  const newName = window.prompt("输入新名称：", currentName);
  if (!newName || !newName.trim() || newName.trim() === currentName) return;
  await api("/api/models/" + modelId + "/name", { method: "PATCH", body: { name: newName.trim() } });
  toast("模型已改名", "success");
  await refreshModels();
}

async function deleteModelById(modelId) {
  const meta = state.modelMeta[modelId];
  const name = ensureModelFilenameExtension(meta?.name || modelId, meta?.model_type);
  if (!window.confirm(`确定删除模型「${name}」吗？`)) return;
  await api(`/api/models/${modelId}`, { method: "DELETE" });
  toast("模型已删除", "success");
  await refreshModels();
}

async function downloadModelById(modelId) {
  if (!modelId) { toast("请选择模型", "warning"); return; }
  try {
    const resp = await fetch(`/api/models/${modelId}/download`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    const a = document.createElement("a");
    const meta = state.modelMeta[modelId] || {};
    const cd = resp.headers.get("content-disposition") || "";
    const filenameStar = cd.match(/(?:^|;)\s*filename\*=UTF-8''([^;]+)/i);
    const match = cd.match(/(?:^|;)\s*filename="?([^";]+)"?/i);
    const headerName = filenameStar
      ? decodeURIComponent(filenameStar[1])
      : (match ? decodeURIComponent(match[1]) : "");
    a.href = URL.createObjectURL(blob);
    a.download = ensureModelFilenameExtension(headerName || meta.name || "model", meta.model_type);
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    toast("下载开始", "success");
  } catch (e) {
    toast(e.message, "error");
  }
}

async function handleResourceAction(btn) {
  const kind = btn.dataset.kind || "";
  const action = btn.dataset.action || "";
  const id = btn.dataset.id || "";
  if (kind === "imageset") {
    if (action === "view") return openResourceImageset(id);
    if (action === "rename") return renameImagesetById(id);
    if (action === "assign") return openAssignModal(id);
    if (action === "download") return downloadImagesetById(id);
    if (action === "delete") return deleteImagesetById(id);
  }
  if (kind === "model") {
    if (action === "use") {
      setActivePanel("panel-annotate");
      setSelectValue("modelSelect", id);
      setMappingConfirmed(false, "已从资源中心选择权重：请读取类别并确认映射。");
      return loadClasses().catch(() => {});
    }
    if (action === "rename") return renameModelById(id);
    if (action === "download") return downloadModelById(id);
    if (action === "delete") return deleteModelById(id);
  }
}


/* ========== Model Upload ========== */
async function uploadModel() {
  const file = $("modelFile").files[0];
  if (!file) { toast("请选择模型文件"); return; }
  const form = new FormData();
  form.append("model_file", file);
  const classes = $("classesFile").files[0];
  if (classes) form.append("classes_file", classes);
  $("modelUploadStatus").textContent = "上传中...";
  try {
    const data = await uploadFormWithProgress("/api/models/upload", {
      method: "POST",
      body: form,
      statusEl: "modelUploadStatus",
      label: "上传模型权重",
      processingText: "上传完成，正在读取模型类别...",
    });
    $("modelUploadStatus").textContent = `模型保存成功: ${data.name}，类别数 ${data.classes.length}`;
    await refreshModels();
    setSelectValue("modelSelect", data.model_id);
    setSelectValue("trainSystemModelSelect", data.model_id);
    await loadClasses();
  } catch (e) {
    $("modelUploadStatus").textContent = e.message;
  }
}

/* ========== Video Extract ========== */
async function uploadVideoAndExtract() {
  const file = $("videoFile").files[0];
  if (!file) {
    toast("请选择视频文件", "warning");
    $("extractStatus").textContent = "请选择视频文件";
    return;
  }
  const validation = validateVideoFile(file);
  if (!validation.ok) {
    toast(validation.message, "warning");
    $("extractStatus").textContent = validation.message;
    return;
  }
  const sec = Number($("sampleSeconds").value || "1");
  const imagesetName = $("extractImagesetName").value.trim();
  $("extractStatus").textContent = "上传视频中...";
  try {
    const form = new FormData();
    form.append("file", file);
    const uploaded = await uploadFormWithProgress("/api/videos/upload", {
      method: "POST",
      body: form,
      statusEl: "extractStatus",
      label: "上传视频",
      processingText: "上传完成，正在登记视频...",
    });
    $("extractStatus").textContent = "创建抽帧任务...";
    const job = await api("/api/extract/jobs", {
      method: "POST",
      body: {
        video_id: uploaded.video_id,
        sample_every_seconds: sec,
        imageset_name: imagesetName || null,
        operator: $("extractOperator").value.trim() || "anonymous",
      },
    });
    state.currentExtractJob = job.id;
    pollExtractJob(job.id);
  } catch (e) {
    $("extractStatus").textContent = e.message;
  }
}

async function pollExtractJob(jobId) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/extract/jobs/${jobId}`);
      $("extractStatus").textContent = `抽帧 [${job.status}] ${(job.progress * 100).toFixed(1)}% ${job.progress_text || ""}`;
      if (job.status === "succeeded") {
        clearInterval(timer);
        toast("抽帧完成", "success");
        state.currentExtractJob = null;
        $("extractStatus").textContent = "抽帧完成，可开始下一轮。";
        await refreshImagesetSelects();
        const setId = job.result.imageset_id;
        setSelectValue("galleryImageset", setId);
        setSelectValue("annotateImageset", setId);
        setSelectValue("aiImagesetSelect", setId);
        await loadGallery();
        await refreshHistoryPanel();
      }
      if (job.status === "failed") {
        clearInterval(timer);
        $("extractStatus").textContent = `失败: ${job.error}`;
      }
      if (job.status === "cancelled") {
        clearInterval(timer);
        $("extractStatus").textContent = `已取消: ${job.error || ""}`;
      }
    } catch (e) {
      clearInterval(timer);
      $("extractStatus").textContent = e.message;
    }
  }, 1200);
}


/* ========== Folder Upload ========== */
function updateAnnotateSourceMode() {
  const mode = $("annotateSourceMode")?.value || "imageset";
  const isImageset = mode === "imageset";
  const imagesetBlock = $("annotateSourceImageset");
  const folderBlock = $("annotateSourceFolder");
  const folderActions = $("annotateSourceFolderActions");
  if (imagesetBlock) imagesetBlock.style.display = isImageset ? "" : "none";
  if (folderBlock) folderBlock.style.display = isImageset ? "none" : "";
  if (folderActions) folderActions.style.display = isImageset ? "none" : "";
}

function updateAiSourceMode() {
  const mode = $("aiSourceMode")?.value || "imageset";
  const isImageset = mode === "imageset";
  if ($("aiSourceImageset")) $("aiSourceImageset").style.display = isImageset ? "" : "none";
  if ($("aiSourceFolder")) $("aiSourceFolder").style.display = isImageset ? "none" : "";
  if ($("aiSourceFolderActions")) $("aiSourceFolderActions").style.display = isImageset ? "none" : "";
}


/* ========== Model Delete ========== */
async function renameCurrentModel() {
  return renameModelFromSelect("modelSelect");
}

async function renameModelFromSelect(selectId) {
  const sel = $(selectId);
  const modelId = sel?.value;
  if (!modelId) { toast("请先选择模型", "warning"); return; }
  const currentName = state.modelMeta[modelId]?.name || sel.options[sel.selectedIndex]?.textContent?.split(" [")[0] || "";
  const newName = window.prompt("输入新名称：", currentName);
  if (!newName || !newName.trim()) return;
  try {
    await api("/api/models/" + modelId + "/name", {
      method: "PATCH",
      body: { name: newName.trim() },
    });
    toast("模型已改名", "success");
    await refreshModels();
  } catch (err) {
    toast("改名失败: " + err.message, "error");
  }
}

async function deleteCurrentModel() {
  const modelId = $("modelSelect").value;
  if (!modelId) { toast("请先选择模型"); return; }
  if (!window.confirm("确定删除当前模型吗？")) return;
  await api(`/api/models/${modelId}`, { method: "DELETE" });
  toast("模型已删除", "success");
  await refreshModels();
  state.sourceClasses = [];
  state.targetClasses = [];
  $("classChecklist").innerHTML = "";
  updateTargetClassesStatus();
  setMappingConfirmed(false, "模型已删除：请重新选择模型并确认映射。");
}




async function cancelCurrentExtract() { await cancelJobById(state.currentExtractJob, "抽帧", "extractStatus"); }


/* ========== Hotkeys ========== */
function bindHotkeys() {
  document.addEventListener("keydown", async (ev) => {
    if (isTypingTarget(ev.target)) return;
    const key = String(ev.key || "").toLowerCase();
    const hasModifier = ev.ctrlKey || ev.metaKey || ev.altKey;
    const videoActive = $("panel-video")?.classList.contains("active");
    const annotateActive = $("panel-annotate")?.classList.contains("active");
    const qwenActive = $("panel-qwen")?.classList.contains("active");

    if (videoActive && ev.key === "[") { ev.preventDefault(); galleryPrevPage(); return; }
    if (videoActive && ev.key === "]") { ev.preventDefault(); galleryNextPage(); return; }
    if (videoActive && !hasModifier && key === "a") { ev.preventDefault(); toggleSelectAll(); return; }
    const refineActive = $("panel-refine")?.classList.contains("active");
    if (videoActive && ev.key === "Delete") { ev.preventDefault(); await batchDelete(); return; }
    if (refineActive && ev.key === "Delete") { ev.preventDefault(); _refineDeleteSelectedBox(); return; }
    if (refineActive && (ev.ctrlKey || ev.metaKey) && key === "s") { ev.preventDefault(); await saveRefineCurrentImage(); return; }
    if (refineActive && !hasModifier && !state.refineInteraction) {
      if (key === "q") { ev.preventDefault(); _refineDeleteSelectedBox(); return; }
      if (key === "w") { ev.preventDefault(); _refineSetCreateMode(true); return; }
      if (key === "s" && state.refineCreateMode) { ev.preventDefault(); _refineSetCreateMode(false); return; }
      if (key === "a") {
        ev.preventDefault();
        if (state.refineCreateMode) _refineStepDefaultClass(-1);
        else await _refineMoveWithAutoSave(-1);
        return;
      }
      if (key === "d") {
        ev.preventDefault();
        if (state.refineCreateMode) _refineStepDefaultClass(1);
        else await _refineMoveWithAutoSave(1);
        return;
      }
    }
    // 方向键快速翻图：自动保存当前张再切换
    if (refineActive && ev.key === "ArrowLeft") { ev.preventDefault(); await _refineMoveWithAutoSave(-1); return; }
    if (refineActive && ev.key === "ArrowRight") { ev.preventDefault(); await _refineMoveWithAutoSave(1); return; }
    if (annotateActive && (ev.ctrlKey || ev.metaKey) && ev.key === "Enter") { ev.preventDefault(); await startAnnotate(); return; }
    if (qwenActive && (ev.ctrlKey || ev.metaKey) && ev.key === "Enter") { ev.preventDefault(); await qwenAutoAnnotate(); }
  });
}

/* ========== Event Binding ========== */



function bindEvents() {
  // 资源中心：所有“资源从哪里来、怎么维护”的入口集中在这里。
  on("btnPickResourceFolder", "click", async () => { await openNativeFilePicker("resourceFolderInput"); });
  on("resourceFolderInput", "change", syncResourceFolderUi);
  on("btnResourceUploadFolder", "click", uploadResourceFolderAsImageset);
  on("btnRefreshResourceImagesets", "click", refreshImagesetSelects);
  on("btnRefreshResourceModels", "click", refreshModels);
  on("btnPickModelFile", "click", async () => { await openNativeFilePicker("modelFile"); });
  on("btnPickClassesFile", "click", async () => { await openNativeFilePicker("classesFile"); });
  on("modelFile", "change", syncModelUploadUi);
  on("classesFile", "change", syncModelUploadUi);
  if ($("resourceImagesetList")) {
    $("resourceImagesetList").addEventListener("click", async (ev) => {
      const btn = ev.target?.closest?.(".resource-action");
      if (btn) await handleResourceAction(btn);
    });
  }
  if ($("resourceModelList")) {
    $("resourceModelList").addEventListener("click", async (ev) => {
      const btn = ev.target?.closest?.(".resource-action");
      if (btn) await handleResourceAction(btn);
    });
  }

  on("btnPickVideoFile", "click", async () => { await openNativeFilePicker("videoFile"); });
  on("btnUploadExtract", "click", uploadVideoAndExtract);
  on("btnCancelExtract", "click", cancelCurrentExtract);
  on("videoFile", "change", syncVideoFileUi);
  on("btnRefreshImagesets", "click", refreshImagesetSelects);
  on("galleryImageset", "change", async () => { state.galleryPage = 1; await loadGallery(); });
  on("btnLoadGallery", "click", async () => { state.galleryPage = 1; await loadGallery(); });
  on("galleryHasLabelFilter", "change", async () => { state.galleryPage = 1; await loadGallery(); });
  on("galleryKeyword", "change", async () => { state.galleryPage = 1; await loadGallery(); });
  on("btnGalleryPrev", "click", galleryPrevPage);
  on("btnGalleryNext", "click", galleryNextPage);
  on("btnRenameImageset", "click", renameCurrentImageset);
  on("btnAssignImageset", "click", assignCurrentImageset);
  on("btnDeleteImageset", "click", deleteCurrentImageset);
  on("btnSelectAll", "click", toggleSelectAll);
  on("btnBatchDelete", "click", batchDelete);

  on("btnUploadModel", "click", uploadModel);
  on("btnRefreshModels", "click", refreshModels);
  on("btnLoadClasses", "click", loadClasses);
  on("btnRenameModel", "click", renameCurrentModel);
  on("btnDownloadSelectedModel", "click", () => downloadModelFromSelect("modelSelect"));
  on("btnDeleteModel", "click", deleteCurrentModel);
  on("btnClassAll", "click", () => setClassCheckAll(true));
  on("btnClassNone", "click", () => setClassCheckAll(false));
  on("btnDefaultMapping", "click", defaultMapping);
  on("btnApplyTargetClasses", "click", applyManualTargetClasses);
  on("btnClearTargetClasses", "click", clearTargetClasses);
  on("btnConfirmClassMapping", "click", confirmClassMapping);
  on("btnSaveClassPreset", "click", saveClassPreset);
  on("btnLoadClassPreset", "click", applyClassPreset);
  on("btnDeleteClassPreset", "click", deleteClassPreset);
  on("modelSelect", "change", () => {
    setMappingConfirmed(false, "模型已切换：请重新加载类别并确认映射。");
  });
  on("classPresetSelect", "change", () => {
    const presetId = $("classPresetSelect")?.value || "";
    const preset = state.classPresets.find((x) => x.preset_id === presetId);
    if ($("classPresetName")) $("classPresetName").value = preset?.name || "";
  });
  const checklist = $("classChecklist");
  if (checklist) {
    checklist.addEventListener("change", () => {
      setMappingConfirmed(false, "映射已变更：请重新点击\"确认映射（必做）\"。");
    });
  }
  on("btnRefreshAnnotateImagesets", "click", async () => { await refreshImagesetSelects(); loadInlinePreview($("annotateImageset")?.value, "annotateInlinePreview"); });
  on("annotateImageset", "change", async () => {
    await refreshImagesetSelects();
    loadInlinePreview($("annotateImageset")?.value, "annotateInlinePreview");
    if (state.sourceClasses.length) await renderClassChecklist();
  });
  on("btnUploadFolder", "click", uploadFolderAsImageset);
  on("btnStartAnnotate", "click", startAnnotate);
  on("btnCancelAnnotate", "click", cancelCurrentAnnotate);
  on("btnRollbackCurrentAnnotate", "click", rollbackCurrentAnnotate);

  on("btnPreviewAnnotateImageset", "click", () => openImagesetPreview($("annotateImageset")?.value));
  on("btnRenameAnnotateImageset", "click", () => renameImagesetFromSelect("annotateImageset"));
  on("btnDownloadAnnotateImageset", "click", () => downloadImageset("annotateImageset"));
  on("btnPreviewAiImageset", "click", () => openImagesetPreview($("aiImagesetSelect")?.value));
  on("btnRenameAiImageset", "click", () => renameImagesetFromSelect("aiImagesetSelect"));
  on("btnDownloadAiImageset", "click", () => downloadImageset("aiImagesetSelect"));
  on("btnCloseImagesetPreview", "click", closeImagesetPreview);
  on("btnPreviewModalPrev", "click", () => { _previewModal.page--; loadPreviewModal(); });
  on("btnPreviewModalNext", "click", () => { _previewModal.page++; loadPreviewModal(); });
  on("previewModalFilter", "change", () => { _previewModal.page = 1; loadPreviewModal(); });
  if ($("imagesetPreviewModal")) {
    $("imagesetPreviewModal").addEventListener("click", (e) => {
      if (e.target === $("imagesetPreviewModal")) closeImagesetPreview();
    });
  }

  on("btnRefreshAiResources", "click", async () => { await refreshImagesetSelects(); loadInlinePreview($("aiImagesetSelect")?.value, "aiInlinePreview"); });
  on("aiImagesetSelect", "change", () => loadInlinePreview($("aiImagesetSelect")?.value, "aiInlinePreview"));
  on("btnAiUploadFolder", "click", uploadAiFolderAsImageset);
  on("btnDownloadImageset", "click", () => downloadImageset("galleryImageset"));
  on("btnPickQwenRefs", "click", async () => { await openNativeFilePicker("qwenRefImages"); });
  on("qwenRefImages", "change", syncQwenRefFilesUi);
  on("btnQwenAutoAnnotate", "click", qwenAutoAnnotate);
  on("qwenPrecisionMode", "change", syncQwenPrecisionModeUi);
  on("btnCancelQwen", "click", cancelCurrentQwen);
  on("btnRollbackCurrentQwen", "click", rollbackCurrentQwen);

  on("btnRefreshHistory", "click", refreshHistoryPanel);
  on("historyJobType", "change", refreshHistoryPanel);
  on("btnSelectAllHistory", "click", toggleSelectAllHistory);
  on("btnBatchDeleteHistory", "click", batchDeleteHistory);

  on("btnToggleBottomPanel", "click", toggleBottomPanel);
  on("btnUserMenu", "click", toggleUserMenu);
  on("taskCreateAssign", "click", () => openAssignModal($("galleryImageset")?.value || ""));
  on("taskRefreshBoard", "click", async () => { await refreshImagesetSelects(); await refreshTaskLedger(); toast("任务已刷新", "success"); });
  on("btnRefreshTaskLedger", "click", async () => { await refreshTaskLedger(); toast("流水已刷新", "success"); });
  on("btnClearTaskLedger", "click", clearTaskLedger);
  if ($("taskBoard")) {
    $("taskBoard").addEventListener("click", async (ev) => {
      const btn = ev.target?.closest?.(".task-action");
      if (!btn) return;
      await openTaskImageset(btn.dataset.imagesetId || "", btn.dataset.action || "", btn.dataset.assigneeId || "");
    });
  }
  on("btnLogout", "click", logout);
  on("btnChangePassword", "click", openPasswordModal);
  on("btnClosePassword", "click", closePasswordModal);
  on("btnSubmitPassword", "click", submitPasswordChange);
  on("btnCloseAssign", "click", closeAssignModal);
  on("btnSubmitAssign", "click", submitAssignModal);
  on("assignImagesetSelect", "change", async () => { await renderAssignOperators(); });

  // Remap panel
  on("btnRefreshRemapImagesets", "click", refreshImagesetSelects);
  on("btnLoadClassMapping", "click", loadClassMapping);
  on("btnRenameRemapImageset", "click", () => renameImagesetFromSelect("remapImageset"));
  on("btnDownloadRemapImageset", "click", () => downloadImageset("remapImageset"));
  on("btnApplyRemap", "click", applyRemap);
  on("btnResetRemap", "click", loadClassMapping);

  // Refine panel
  on("btnRefreshRefineImagesets", "click", async () => {
    await refreshImagesetSelects();
    await loadRefineImageset({ preserveImageId: state.refineCurrentImageId, skipDirty: true });
  });
  on("refineImageset", "change", async () => {
    const prevValue = state.refineCurrentImageId;
    await loadRefineImageset({ preserveImageId: prevValue });
  });
  on("refineImageSelect", "change", async () => {
    const nextId = $("refineImageSelect")?.value || "";
    const ok = await loadRefineImage(nextId);
    if (!ok && $("refineImageSelect")) $("refineImageSelect").value = state.refineCurrentImageId;
    else _refineFocusWorkspace();
  });
  on("btnRefinePrevImage", "click", () => _refineMoveImage(-1));
  on("btnRefineNextImage", "click", () => _refineMoveImage(1));
  on("btnPreviewRefineImageset", "click", () => openImagesetPreview($("refineImageset")?.value));
  on("btnRenameRefineImageset", "click", () => renameImagesetFromSelect("refineImageset"));
  on("btnDownloadRefineImageset", "click", () => downloadImageset("refineImageset"));
  on("btnRefineCreateBox", "click", () => {
    _refineSetCreateMode(!state.refineCreateMode);
    _refineFocusWorkspace();
  });
  on("btnRefineCreateClass", "click", _refineCreateOrAssignClass);
  on("btnRefineDeleteSelected", "click", _refineDeleteSelectedBox);
  on("btnRefineSave", "click", saveRefineCurrentImage);
  on("btnRefineRollback", "click", rollbackRefineCurrentImage);
  on("refineNewBoxClass", "change", () => {
    state.refineDefaultClassValue = $("refineNewBoxClass")?.value || "";
    _refineFocusWorkspace();
  });
  on("refineShapeType", "change", () => {
    const sel = $("refineShapeType");
    if (!sel) return;
    const nextTask = normalizeLabelTask(sel.value || "detect");
    const lockedTask = _refineLockedShapeTask();
    if (lockedTask && nextTask !== lockedTask) {
      sel.value = lockedTask;
      toast(`当前图片集已固定为 ${labelTaskLabel(lockedTask)}，不能混用`, "warning");
      return;
    }
    if (state.refineImageData && !lockedTask) {
      state.refineImageData.label_task = nextTask;
    }
    state.refineDraftPoints = [];
    state.refineDraftHoverPoint = null;
    renderRefineOverlay();
    _refineSetImageStatus(`新增形状类型：${labelTaskLabel(nextTask)}`);
    _refineFocusWorkspace();
  });
  on("refineSelectedClass", "change", () => {
    _refineApplyClassToSelected($("refineSelectedClass")?.value || "");
    _refineFocusWorkspace();
  });
  on("refineImage", "load", renderRefineOverlay);
  // box 改用 px 相对 img 实际位置定位，窗口变化时需要重绘
  window.addEventListener("resize", function() {
    if (state.refineImageData) renderRefineOverlay();
  });
  if (window.ResizeObserver && $("refineStage")) {
    const refineResizeObserver = new ResizeObserver(() => {
      if (state.refineImageData && $("panel-refine")?.classList.contains("active")) {
        renderRefineOverlay();
      }
    });
    refineResizeObserver.observe($("refineStage"));
    if ($("refineImage")) refineResizeObserver.observe($("refineImage"));
  }
  if ($("refineOverlay")) {
    $("refineOverlay").addEventListener("mousedown", (ev) => {
      if (ev.button !== 0) return;
      if (!state.refineCreateMode) return;
      ev.preventDefault();
      _refineBeginCreate(ev);
    });
    $("refineOverlay").addEventListener("mouseleave", () => {
      if (!state.refineCreateMode || state.refineInteraction) return;
      if (!state.refineDraftHoverPoint) return;
      state.refineDraftHoverPoint = null;
      renderRefineOverlay();
    });
  }
  window.addEventListener("mousemove", _refineHandlePointerMove);
  window.addEventListener("mouseup", _refineHandlePointerUp);

  // Training panel
  on("btnRefreshTrainImagesets", "click", async () => {
    await refreshImagesetSelects();
    updateTrainDatasetInfo();
    resetTrainArtifacts();
    loadInlinePreview($("trainImageset")?.value, "trainInlinePreview", { collapsed: true });
  });
  on("trainImageset", "change", async () => {
    const meta = state.imagesetMeta[$("trainImageset")?.value || ""];
    if (meta?.label_task && $("trainTask")) $("trainTask").value = normalizeLabelTask(meta.label_task);
    syncTrainTaskUi(true);
    updateTrainDatasetInfo();
    resetTrainArtifacts();
    loadInlinePreview($("trainImageset")?.value, "trainInlinePreview", { collapsed: true });
  });
  on("trainTask", "change", () => syncTrainTaskUi(true));
  on("trainBaseModelSelect", "change", syncTrainBaseModelUI);
  on("btnPreviewTrainImageset", "click", () => openImagesetPreview($("trainImageset")?.value));
  on("btnRenameTrainImageset", "click", () => renameImagesetFromSelect("trainImageset"));
  on("btnDownloadTrainImageset", "click", () => downloadImageset("trainImageset"));
  on("btnStartTrain", "click", startTrain);
  on("btnCancelTrain", "click", cancelCurrentTrain);
  on("btnRefreshDownloadModels", "click", refreshModels);
  on("btnDownloadModel", "click", downloadModel);
  on("btnRenameDownloadModel", "click", () => renameModelFromSelect("downloadModelSelect"));
  on("btnRenameTrainSystemModel", "click", () => renameModelFromSelect("trainSystemModelSelect"));
  on("btnDownloadTrainSystemModel", "click", () => downloadModelFromSelect("trainSystemModelSelect"));
}


function bindDragDrop() {
  document.querySelectorAll(".file-picker").forEach(picker => {
    const input = picker.querySelector("input[type=file]");
    if (!input) return;
    picker.addEventListener("dragover", e => { e.preventDefault(); picker.classList.add("drag-over"); });
    picker.addEventListener("dragleave", () => picker.classList.remove("drag-over"));
    picker.addEventListener("drop", e => {
      e.preventDefault();
      picker.classList.remove("drag-over");
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
  });
}

/* ========== Bootstrap ========== */
async function bootstrap() {
  bindSectionNav();
  initBottomPanel();
  bindEvents();
  bindHotkeys();
  bindDragDrop();
  loadClassPresets();
  await loadSessionContext();
  await checkHealth();
  await Promise.all([refreshImagesetSelects(), refreshModels()]);
  if ($("galleryImageset")?.value) {
    state.galleryPage = 1;
    await loadGallery().catch(() => {});
  }
  updateAnnotateSourceMode();
  updateAiSourceMode();
  loadInlinePreview($("annotateImageset")?.value, "annotateInlinePreview");
  loadInlinePreview($("aiImagesetSelect")?.value, "aiInlinePreview");
  syncQwenPrecisionModeUi();
  syncTrainBaseModelUI();
  updateTrainDatasetInfo();
  loadInlinePreview($("trainImageset")?.value, "trainInlinePreview", { collapsed: true });
  await loadRefineImageset({ skipDirty: true, quiet: true }).catch(() => {});
  updateTargetClassesStatus();
  setMappingConfirmed(false, "请先确认类别映射，确认后才可启动权重自动打标。");
  syncVideoFileUi();
  syncResourceFolderUi();
  syncModelUploadUi();
  syncQwenRefFilesUi();
  await loadClasses().catch(() => {});
  refreshJobsPanel();
  refreshHistoryPanel();
  refreshTaskLedger();
  setInterval(refreshJobsPanel, 2500);
  setInterval(refreshHistoryPanel, 4000);
  setInterval(refreshTaskLedger, 6000);
}

/* ========== Lightbox 图片放大 ========== */
function openLightbox(src) {
  var box = $("lightbox");
  var img = $("lightboxImg");
  if (!box || !img || !src) return;
  img.src = src;
  box.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeLightbox(ev) {
  // 点击图片本身不关闭（方便拖拽查看），点背景关闭
  if (ev && ev.target && ev.target.id === "lightboxImg") return;
  var box = $("lightbox");
  if (!box) return;
  box.classList.add("hidden");
  $("lightboxImg").src = "";
  document.body.style.overflow = "";
}

// ESC 关闭
document.addEventListener("keydown", function(ev) {
  if (ev.key === "Escape") closeLightbox();
});

// 全局事件委托：点击带 cursor:zoom-in 的图片自动放大
document.addEventListener("click", function(ev) {
  var target = ev.target;
  if (target.tagName !== "IMG") return;
  // 精修区的图片不走 lightbox（它有自己的交互）
  if (target.id === "refineImage") return;
  var style = window.getComputedStyle(target);
  if (style.cursor === "zoom-in") {
    ev.preventDefault();
    ev.stopPropagation();
    openLightbox(target.src);
  }
});

bootstrap();
