async function refreshGalleryFilterHint(imagesetId, filteredTotal) {
  const hint = $("galleryFilterHint");
  if (!hint || !imagesetId) return;
  try {
    const meta = state.imagesetMeta[imagesetId];
    const total = Number(meta?.image_count || 0);
    const withLabel = Number(meta?.label_count || 0);
    const withoutLabel = Math.max(0, total - withLabel);
    hint.textContent = `筛选命中 ${filteredTotal} / 总 ${total}；有txt ${withLabel}，无txt ${withoutLabel}`;
  } catch (e) {
    hint.textContent = `筛选统计读取失败: ${e.message}`;
  }
}

/* ========== Gallery ========== */
async function loadGallery() {
  const imagesetId = $("galleryImageset").value;
  if (!imagesetId) { toast("请选择图片集"); return; }
  try {
    const q = new URLSearchParams();
    q.set("page", String(Math.max(1, state.galleryPage || 1)));
    q.set("page_size", String(Math.max(1, state.galleryPageSize || 120)));
    const hasLabel = $("galleryHasLabelFilter")?.value || "";
    const keyword = $("galleryKeyword")?.value?.trim() || "";
    if (hasLabel) q.set("has_label", hasLabel);
    if (keyword) q.set("keyword", keyword);
    const data = await api(`/api/imagesets/${imagesetId}/images?${q.toString()}`);
    state.currentGalleryImages = data.items;
    state.galleryTotal = Number(data.total || data.items.length || 0);
    state.galleryPage = Number(data.page || state.galleryPage || 1);
    state.galleryPageSize = Number(data.page_size || state.galleryPageSize || 120);
    renderGallery(data.items);
    const labeled = data.items.filter((x) => x.label_exists).length;
    const maxPage = Math.max(1, Math.ceil(state.galleryTotal / Math.max(1, state.galleryPageSize)));
    $("galleryStatus").textContent = `当前页 ${data.items.length} 张，有标注 ${labeled}；总 ${state.galleryTotal}`;
    await refreshGalleryFilterHint(imagesetId, state.galleryTotal);
    if ($("galleryPagerInfo")) $("galleryPagerInfo").textContent = `第 ${state.galleryPage} / ${maxPage} 页`;
    if ($("btnGalleryPrev")) $("btnGalleryPrev").disabled = state.galleryPage <= 1;
    if ($("btnGalleryNext")) $("btnGalleryNext").disabled = state.galleryPage >= maxPage;
  } catch (e) {
    $("galleryStatus").textContent = e.message;
  }
}

async function deleteCurrentImageset() {
  const imagesetId = $("galleryImageset").value;
  if (!imagesetId) { toast("请先选择图片集"); return; }
  if (!window.confirm("确定删除当前图片集及其图片吗？")) return;
  await api(`/api/imagesets/${imagesetId}`, { method: "DELETE" });
  toast("图片集已删除", "success");
  await refreshImagesetSelects();
  $("galleryGrid").innerHTML = "";
}

async function renameCurrentImageset() {
  return renameImagesetFromSelect("galleryImageset");
}

async function renameImagesetFromSelect(selectId) {
  const imagesetId = $(selectId)?.value;
  if (!imagesetId) { toast("请先选择图片集"); return; }
  const meta = state.imagesetMeta[imagesetId];
  const oldName = meta?.name || "";
  const newName = window.prompt("输入新名称:", oldName);
  if (!newName || newName.trim() === oldName) return;
  const form = new FormData();
  form.append("name", newName.trim());
  await api(`/api/imagesets/${imagesetId}/rename`, { method: "PATCH", body: form });
  toast("已重命名", "success");
  await refreshImagesetSelects();
}

function renderGallery(items) {
  const box = $("galleryGrid");
  box.innerHTML = "";
  if (!items.length) {
    box.innerHTML = '<div class="status-box">当前筛选条件下没有图片</div>';
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "image-card";
    const deleteButton = isAdminUser()
      ? `<button class="danger btn-delete" data-id="${item.image_id}">删</button>`
      : "";
    card.innerHTML = `
      <div class="head">
        <label><input type="checkbox" class="img-check" data-id="${item.image_id}"> 选</label>
        ${deleteButton}
      </div>
      <img src="${item.url}" alt="${item.filename}">
      <div class="meta">
        <span class="image-name" title="${item.filename}">${item.filename}</span>
        <span>${item.label_exists ? "有txt" : "无txt"}</span>
      </div>
    `;
    box.appendChild(card);
  });
  box.querySelectorAll(".btn-delete").forEach((btn) => {
    btn.addEventListener("click", async () => { await deleteSingleImage(btn.dataset.id); });
  });
}

async function deleteSingleImage(imageId) {
  await api(`/api/images/${imageId}`, { method: "DELETE" });
  toast("已删除", "success");
  await loadGallery();
  await refreshImagesetSelects();
}

function toggleSelectAll() {
  const checks = Array.from(document.querySelectorAll(".img-check"));
  if (!checks.length) return;
  const allChecked = checks.every((x) => x.checked);
  checks.forEach((x) => { x.checked = !allChecked; });
}

async function batchDelete() {
  const ids = Array.from(document.querySelectorAll(".img-check:checked")).map((x) => x.dataset.id);
  if (!ids.length) { toast("请先勾选图片"); return; }
  await api("/api/images/delete-batch", { method: "POST", body: { image_ids: ids } });
  toast(`已删除 ${ids.length} 张`, "success");
  await loadGallery();
  await refreshImagesetSelects();
}

function galleryPrevPage() {
  if (state.galleryPage <= 1) return;
  state.galleryPage -= 1;
  loadGallery();
}

function galleryNextPage() {
  const maxPage = Math.max(1, Math.ceil(state.galleryTotal / Math.max(1, state.galleryPageSize)));
  if (state.galleryPage >= maxPage) return;
  state.galleryPage += 1;
  loadGallery();
}

async function downloadImageset(selectId) {
  const imagesetId = $(selectId)?.value;
  if (!imagesetId) { toast("请先选择图片集"); return; }
  window.open(`/api/imagesets/${imagesetId}/download`, "_blank");
}

async function deleteImagesetFromSelect(selectId) {
  const imagesetId = $(selectId)?.value;
  if (!imagesetId) { toast("请先选择图片集"); return; }
  const meta = state.imagesetMeta[imagesetId];
  const name = meta?.name || imagesetId;
  if (!window.confirm(`确定删除数据集「${name}」及其全部图片和标签吗？`)) return;
  try {
    await api(`/api/imagesets/${imagesetId}`, { method: "DELETE" });
    toast("图片集已删除", "success");
    await refreshImagesetSelects();
    await loadGallery().catch(() => {});
  } catch (e) {
    toast(`删除失败: ${e.message}`, "error");
  }
}
