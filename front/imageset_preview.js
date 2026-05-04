var _previewModal = { imagesetId: "", page: 1, pageSize: 24, total: 0 };

function _parseLabelSummary(rawText, classNames) {
  if (!rawText) return "";
  const counts = {};
  for (const line of rawText.split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 5) {
      const cid = parts[0];
      const name = (classNames && classNames[cid]) || `类别${cid}`;
      counts[name] = (counts[name] || 0) + 1;
    }
  }
  if (!Object.keys(counts).length) return "";
  return Object.entries(counts).map(([name, count]) => `${name} ×${count}`).join("，");
}

async function loadInlinePreview(imagesetId, targetId, options = {}) {
  const container = $(targetId);
  if (!container) return;
  if (!imagesetId) { container.innerHTML = ""; return; }
  const collapsed = !!options.collapsed;
  try {
    const data = await api(`/api/imagesets/${imagesetId}/images?page=1&page_size=8`);
    const items = data.items || [];
    const classNames = data.class_names || {};
    state.imagesetClassNames[imagesetId] = classNames;
    const meta = state.imagesetMeta[imagesetId];
    const total = meta?.image_count || data.total || 0;
    const labeled = meta?.label_count || 0;
    const labelTexts = await Promise.all(items.map((item) =>
      item.label_url ? fetchTextNoCache(item.label_url).catch(() => "") : Promise.resolve("")
    ));
    const latestJobId = data.latest_job_id || "";
    const galleryLinks = [];
    if (latestJobId) {
      galleryLinks.push(`<a href="/front/preview.html?job_id=${latestJobId}&prefer_current_labels=1" target="_blank" class="ip-gallery-link">最近结果对照画廊</a>`);
    }
    galleryLinks.push(`<a href="#" class="ip-gallery-link ip-gallery-open-modal" data-imageset-id="${imagesetId}">数据集全图预览</a>`);
    const galleryLink = galleryLinks.join(" ");
    const summaryText = `共 ${total} 张，有标注 ${labeled}，无标注 ${Math.max(0, total - labeled)}`;
    container.innerHTML = `
      <details class="ip-details"${collapsed ? "" : " open"}>
        <summary class="ip-summary">${summaryText} ${galleryLink}${collapsed ? " （点击展开）" : total > 8 ? " （点击收起）" : ""}</summary>
        <div class="ip-grid">${items.map((item, index) => {
          const labelSummary = _parseLabelSummary(labelTexts[index], classNames);
          return `
          <div class="ip-card">
            <img src="${item.url}" alt="${_esc(item.filename)}">
            <div class="ip-meta">
              <span class="ip-name" title="${_esc(item.filename)}">${_esc(item.filename)}</span>
              <span class="ip-label-badge ${item.label_exists ? 'has' : 'none'}">${item.label_exists ? '有标注' : '无标注'}</span>
              ${labelSummary ? `<div class="ip-label-summary">${_esc(labelSummary)}</div>` : ""}
            </div>
          </div>`;
        }).join("")}
        </div>
      </details>`;
    container.querySelectorAll(".ip-gallery-open-modal").forEach((el) => {
      el.addEventListener("click", (event) => {
        event.preventDefault();
        openImagesetPreview(el.dataset.imagesetId);
      });
    });
  } catch {
    container.innerHTML = "";
  }
}

async function openImagesetPreview(imagesetId) {
  if (!imagesetId) { toast("请先选择图片集"); return; }
  _previewModal.imagesetId = imagesetId;
  _previewModal.page = 1;
  $("imagesetPreviewModal").classList.remove("hidden");
  await loadPreviewModal();
}

async function loadPreviewModal() {
  const grid = $("previewModalGrid");
  grid.innerHTML = '<div class="status-box">加载中...</div>';
  try {
    const q = new URLSearchParams();
    q.set("page", String(_previewModal.page));
    q.set("page_size", String(_previewModal.pageSize));
    const filter = $("previewModalFilter")?.value || "";
    if (filter) q.set("has_label", filter);
    const data = await api(`/api/imagesets/${_previewModal.imagesetId}/images?${q}`);
    _previewModal.total = data.total || 0;
    const items = data.items || [];
    const classNames = data.class_names || {};

    const meta = state.imagesetMeta[_previewModal.imagesetId];
    $("previewModalStats").textContent = meta
      ? `${meta.name} -- 图:${meta.image_count} 标:${meta.label_count || 0}`
      : `共 ${_previewModal.total} 张`;

    const labelTexts = await Promise.all(items.map((item) =>
      item.label_url ? fetchTextNoCache(item.label_url).catch(() => "") : Promise.resolve("")
    ));
    grid.innerHTML = items.length ? items.map((item, index) => {
      const labelSummary = _parseLabelSummary(labelTexts[index], classNames);
      return `
      <div class="ip-card">
        <img src="${item.url}" alt="${_esc(item.filename)}">
        <div class="ip-meta">
          <span class="ip-name" title="${_esc(item.filename)}">${_esc(item.filename)}</span>
          <span class="ip-label-badge ${item.label_exists ? "has" : "none"}">${item.label_exists ? "有标注" : "无标注"}</span>
          ${labelSummary ? `<div class="ip-label-summary">${_esc(labelSummary)}</div>` : ""}
          ${labelTexts[index] ? `<div class="ip-txt">${_esc(labelTexts[index])}</div>` : ""}
        </div>
      </div>`;
    }).join("") : '<div class="status-box">无数据</div>';

    const maxPage = Math.max(1, Math.ceil(_previewModal.total / _previewModal.pageSize));
    $("previewModalPager").textContent = `${_previewModal.page} / ${maxPage}`;
    $("btnPreviewModalPrev").disabled = _previewModal.page <= 1;
    $("btnPreviewModalNext").disabled = _previewModal.page >= maxPage;
  } catch (e) {
    grid.innerHTML = `<div class="status-box">${_esc(e.message)}</div>`;
  }
}

function closeImagesetPreview() {
  $("imagesetPreviewModal").classList.add("hidden");
}
