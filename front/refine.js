const REFINE_COLOR_PALETTE = [
  { stroke: "rgba(37,99,235,0.95)", fill: "rgba(37,99,235,0.14)", label: "rgba(30,64,175,0.94)" },
  { stroke: "rgba(16,185,129,0.95)", fill: "rgba(16,185,129,0.14)", label: "rgba(6,95,70,0.94)" },
  { stroke: "rgba(236,72,153,0.95)", fill: "rgba(236,72,153,0.14)", label: "rgba(157,23,77,0.94)" },
];

/* ========== Module 6: Dataset Refine ========== */
const REFINE_NEW_CLASS_PREFIX = "new:";

function _refineStatus(text) {
  const el = $("refineStatus");
  if (el) el.textContent = text || "请选择图片集";
}

function _refineSetImageStatus(note = "") {
  const el = $("refineImageStatus");
  if (!el) return;
  const data = state.refineImageData;
  if (!data) {
    el.textContent = note || "请选择图片";
    return;
  }
  const boxCount = Array.isArray(data.boxes) ? data.boxes.length : 0;
  const dirtyText = state.refineDirty ? "未保存修改" : "已与当前标签同步";
  const modeText = state.refineCreateMode ? "新增框模式" : "选择模式";
  const extra = note ? `，${note}` : "";
  el.textContent = `${data.filename} — ${boxCount} 框，${dirtyText}，${modeText}${extra}`;
}

function _refineFocusWorkspace() {
  const stage = $("refineStage");
  if (!stage || !state.refineImageData) return;
  requestAnimationFrame(() => stage.focus({ preventScroll: true }));
}

function _refineConfirmDiscard() {
  return !state.refineDirty || window.confirm("当前图片有未保存修改，确定丢弃吗？");
}

function _refineLockedShapeTask() {
  const data = state.refineImageData;
  const imagesetId = data?.imageset_id || $("refineImageset")?.value || "";
  const meta = state.imagesetMeta?.[imagesetId] || {};
  const hasBoxes = Array.isArray(data?.boxes) && data.boxes.length > 0;
  const labelCount = Number(meta.label_count || 0);
  if (hasBoxes || labelCount > 0) {
    return normalizeLabelTask(data?.label_task || meta.label_task || "detect");
  }
  return "";
}

function _refineBoxSortKey(box, idx = 0) {
  const classId = box?.class_id;
  const classRank = classId === null || classId === undefined || Number.isNaN(Number(classId))
    ? Number.MAX_SAFE_INTEGER
    : Number(classId);
  const y = Number(box?.y1 || 0);
  const x = Number(box?.x1 || 0);
  return [classRank, y, x, idx];
}

function _refineSortBoxes(boxes) {
  return (boxes || []).slice().sort((a, b) => {
    const ka = _refineBoxSortKey(a);
    const kb = _refineBoxSortKey(b);
    for (let i = 0; i < ka.length; i += 1) {
      if (ka[i] < kb[i]) return -1;
      if (ka[i] > kb[i]) return 1;
    }
    return 0;
  });
}

function _refineNormalizePayload(payload) {
  const normalizedBoxes = Array.isArray(payload?.boxes)
    ? payload.boxes.map((box, idx) => {
        const shapeType = normalizeLabelTask(box.shape_type || payload?.label_task || "detect");
        return {
          box_id: String(box.box_id || `box_${idx}`),
          class_id: Number.isInteger(box.class_id) ? box.class_id : null,
          class_name: String(box.class_name || "").trim(),
          shape_type: shapeType,
          points: shapeType === "detect"
            ? []
            : (Array.isArray(box.points) ? box.points.map((p) => [Number(p?.[0] || 0), Number(p?.[1] || 0)]) : []),
          x1: Number(box.x1 || 0),
          y1: Number(box.y1 || 0),
          x2: Number(box.x2 || 0),
          y2: Number(box.y2 || 0),
        };
      })
    : [];
  return {
    ...payload,
    label_task: normalizeLabelTask(payload?.label_task || "detect"),
    boxes: _refineSortBoxes(normalizedBoxes),
    class_options: Array.isArray(payload?.class_options) ? payload.class_options.map((item) => ({
      class_id: Number(item.class_id),
      name: String(item.name || "").trim(),
    })) : [],
  };
}

function _refinePendingNamesFromBoxes() {
  const boxes = state.refineImageData?.boxes || [];
  const seen = new Set();
  const out = [];
  boxes.forEach((box) => {
    if (box.class_id !== null && box.class_id !== undefined) return;
    const name = String(box.class_name || "").trim();
    if (!name) return;
    const key = name.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push(name);
  });
  return out;
}

function _refineClassOptions() {
  const existing = Array.isArray(state.refineImageData?.class_options) ? state.refineImageData.class_options : [];
  const options = existing.map((item) => ({ class_id: Number(item.class_id), name: String(item.name || "").trim() }));
  const seen = new Set(options.map((item) => item.name.toLowerCase()));
  [...state.refinePendingClassNames, ..._refinePendingNamesFromBoxes()].forEach((name) => {
    const clean = String(name || "").trim();
    if (!clean) return;
    const key = clean.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    options.push({ class_id: null, name: clean });
  });
  return options;
}

function _refineEncodeClassOption(option) {
  return option.class_id === null || option.class_id === undefined
    ? `${REFINE_NEW_CLASS_PREFIX}${encodeURIComponent(option.name)}`
    : `id:${option.class_id}`;
}

function _refineDecodeClassOption(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  if (raw.startsWith(`${REFINE_NEW_CLASS_PREFIX}`)) {
    return { class_id: null, name: decodeURIComponent(raw.slice(REFINE_NEW_CLASS_PREFIX.length)) };
  }
  if (raw.startsWith("id:")) {
    const classId = Number(raw.slice(3));
    const option = _refineClassOptions().find((item) => Number(item.class_id) === classId);
    return { class_id: classId, name: option?.name || `class_${classId}` };
  }
  return null;
}

function _refineClassDisplayName(boxOrOption) {
  const name = String(boxOrOption?.class_name || boxOrOption?.name || "").trim();
  const classId = boxOrOption?.class_id;
  if (classId !== null && classId !== undefined) {
    return name ? `${classId} ${name}` : String(classId);
  }
  if (name) return name;
  return "未命名";
}

function _refineClassColor(boxOrOption) {
  const classId = boxOrOption?.class_id;
  let seed = Number.isFinite(Number(classId)) ? Number(classId) : 0;
  if (classId === null || classId === undefined || Number.isNaN(Number(classId))) {
    const key = _refineClassDisplayName(boxOrOption);
    seed = Array.from(key).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  }
  return REFINE_COLOR_PALETTE[Math.abs(Math.trunc(seed)) % REFINE_COLOR_PALETTE.length];
}

function _refineBoxCoords(box) {
  return `(${Math.round(box.x1)}, ${Math.round(box.y1)}) → (${Math.round(box.x2)}, ${Math.round(box.y2)})`;
}

function _refineShapePoints(box) {
  if (!box) return [];
  const shapeType = normalizeLabelTask(box.shape_type || "detect");
  if (shapeType !== "detect" && Array.isArray(box.points) && box.points.length) {
    return box.points.map((p) => [Number(p[0] || 0), Number(p[1] || 0)]);
  }
  return [
    [Number(box.x1 || 0), Number(box.y1 || 0)],
    [Number(box.x2 || 0), Number(box.y1 || 0)],
    [Number(box.x2 || 0), Number(box.y2 || 0)],
    [Number(box.x1 || 0), Number(box.y2 || 0)],
  ];
}

function _refineMinDraftPoints(shapeType) {
  const task = normalizeLabelTask(shapeType);
  if (task === "segment") return 3;
  if (task === "obb") return 2;
  return 2;
}

function _refineCreateModeHint(shapeType) {
  const task = normalizeLabelTask(shapeType);
  if (task === "detect") return "拖拽画框，A/D 切标签，S 结束新增";
  if (task === "obb") return "OBB：先像普通框一样拖拽创建，选中后拖顶部旋转点可旋转，四点仍可微调，A/D 切标签，S 结束新增";
  return "SEG：连续点击落点实时预览，A/D 切标签，S 完成多边形";
}

function _refineNormalizePoint(point) {
  return [Number(point?.[0] || 0), Number(point?.[1] || 0)];
}

function _refineRectPointsFromDiagonal(start, end) {
  const [sx, sy] = _refineNormalizePoint(start);
  const [ex, ey] = _refineNormalizePoint(end);
  const x1 = Math.min(sx, ex);
  const x2 = Math.max(sx, ex);
  const y1 = Math.min(sy, ey);
  const y2 = Math.max(sy, ey);
  return [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ];
}

function _refineDistanceBetweenPoints(a, b) {
  const [ax, ay] = _refineNormalizePoint(a);
  const [bx, by] = _refineNormalizePoint(b);
  return Math.hypot(bx - ax, by - ay);
}

function _refineObbMetrics(points) {
  const source = _refineNormalizeObbPoints(points || []);
  if (source.length < 4) return null;
  const [p0, p1, p2] = source;
  const center = [
    (source[0][0] + source[2][0]) / 2,
    (source[0][1] + source[2][1]) / 2,
  ];
  const width = Math.max(1e-6, _refineDistanceBetweenPoints(p0, p1));
  const height = Math.max(1e-6, _refineDistanceBetweenPoints(p1, p2));
  const topMid = [
    (p0[0] + p1[0]) / 2,
    (p0[1] + p1[1]) / 2,
  ];
  const topVec = [topMid[0] - center[0], topMid[1] - center[1]];
  const topLen = Math.hypot(topVec[0], topVec[1]) || 1;
  const topUnit = [topVec[0] / topLen, topVec[1] / topLen];
  const downAxis = [-topUnit[0], -topUnit[1]];
  const widthAxis = [downAxis[1], -downAxis[0]];
  return {
    points: source,
    center,
    width,
    height,
    topMid,
    topUnit,
    downAxis,
    widthAxis,
  };
}

function _refineBuildObbPoints(center, width, height, widthAxis, downAxis) {
  const [cx, cy] = _refineNormalizePoint(center);
  const wx = Number(widthAxis?.[0] || 1);
  const wy = Number(widthAxis?.[1] || 0);
  const dx = Number(downAxis?.[0] || 0);
  const dy = Number(downAxis?.[1] || 1);
  const halfW = Math.max(1e-6, Number(width || 1)) / 2;
  const halfH = Math.max(1e-6, Number(height || 1)) / 2;
  return [
    [cx - (wx * halfW) - (dx * halfH), cy - (wy * halfW) - (dy * halfH)],
    [cx + (wx * halfW) - (dx * halfH), cy + (wy * halfW) - (dy * halfH)],
    [cx + (wx * halfW) + (dx * halfH), cy + (wy * halfW) + (dy * halfH)],
    [cx - (wx * halfW) + (dx * halfH), cy - (wy * halfW) + (dy * halfH)],
  ];
}

function _refineRotateObbByPointer(points, pointer) {
  const metrics = _refineObbMetrics(points);
  if (!metrics) return _refineNormalizeObbPoints(points || []);
  const [cx, cy] = metrics.center;
  const vx = Number(pointer?.x || 0) - cx;
  const vy = Number(pointer?.y || 0) - cy;
  const len = Math.hypot(vx, vy);
  if (len < 1e-6) return metrics.points;
  const topUnit = [vx / len, vy / len];
  const downAxis = [-topUnit[0], -topUnit[1]];
  const widthAxis = [downAxis[1], -downAxis[0]];
  return _refineBuildObbPoints(metrics.center, metrics.width, metrics.height, widthAxis, downAxis);
}

function _refineObbCornerSigns(pointIndex) {
  const signs = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ];
  return signs[Number(pointIndex) || 0] || signs[0];
}

function _refineResizeObbFromCorner(points, pointIndex, pointer) {
  const metrics = _refineObbMetrics(points);
  if (!metrics) return _refineNormalizeObbPoints(points || []);
  const cornerIndex = Number(pointIndex) || 0;
  const oppositeIndex = (cornerIndex + 2) % 4;
  const fixedCorner = metrics.points[oppositeIndex];
  const [signW, signD] = _refineObbCornerSigns(cornerIndex);
  const vector = [
    Number(pointer?.x || 0) - fixedCorner[0],
    Number(pointer?.y || 0) - fixedCorner[1],
  ];
  const widthProjection = (vector[0] * metrics.widthAxis[0]) + (vector[1] * metrics.widthAxis[1]);
  const heightProjection = (vector[0] * metrics.downAxis[0]) + (vector[1] * metrics.downAxis[1]);
  const width = Math.max(1, signW * widthProjection);
  const height = Math.max(1, signD * heightProjection);
  const center = [
    fixedCorner[0] + ((signW * width * metrics.widthAxis[0]) + (signD * height * metrics.downAxis[0])) / 2,
    fixedCorner[1] + ((signW * width * metrics.widthAxis[1]) + (signD * height * metrics.downAxis[1])) / 2,
  ];
  return _refineBuildObbPoints(center, width, height, metrics.widthAxis, metrics.downAxis);
}

function _refineSamePoint(a, b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return Math.abs(Number(a.x) - Number(b.x)) < 0.5 && Math.abs(Number(a.y) - Number(b.y)) < 0.5;
}

function _refineNormalizeObbPoints(points) {
  const pts = (points || []).map(_refineNormalizePoint);
  if (pts.length < 2) return pts;
  if (pts.length === 2) return _refineRectPointsFromDiagonal(pts[0], pts[1]);
  const source = pts.slice(0, 4);
  const cx = source.reduce((sum, p) => sum + p[0], 0) / source.length;
  const cy = source.reduce((sum, p) => sum + p[1], 0) / source.length;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  source.forEach((point) => {
    const dx = point[0] - cx;
    const dy = point[1] - cy;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  });
  let theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  if (!Number.isFinite(theta)) theta = 0;
  let ux = Math.cos(theta);
  let uy = Math.sin(theta);
  if (Math.hypot(ux, uy) < 1e-6) {
    const edge = [source[1][0] - source[0][0], source[1][1] - source[0][1]];
    const edgeLen = Math.hypot(edge[0], edge[1]) || 1;
    ux = edge[0] / edgeLen;
    uy = edge[1] / edgeLen;
  }
  const vx = -uy;
  const vy = ux;
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  source.forEach((point) => {
    const relX = point[0] - cx;
    const relY = point[1] - cy;
    const pu = (relX * ux) + (relY * uy);
    const pv = (relX * vx) + (relY * vy);
    minU = Math.min(minU, pu);
    maxU = Math.max(maxU, pu);
    minV = Math.min(minV, pv);
    maxV = Math.max(maxV, pv);
  });
  return [
    [minU, minV],
    [maxU, minV],
    [maxU, maxV],
    [minU, maxV],
  ].map(([pu, pv]) => [
    cx + (pu * ux) + (pv * vx),
    cy + (pu * uy) + (pv * vy),
  ]);
}

function _refineSyncBoxBounds(box) {
  const shapeType = normalizeLabelTask(box?.shape_type || "detect");
  if (shapeType === "detect") {
    box.points = [];
  }
  let points = _refineShapePoints(box);
  if (shapeType === "obb" && points.length >= 3) {
    points = _refineNormalizeObbPoints(points);
  }
  if (!points.length) return;
  const xs = points.map((p) => Number(p[0] || 0));
  const ys = points.map((p) => Number(p[1] || 0));
  box.x1 = Math.min(...xs);
  box.y1 = Math.min(...ys);
  box.x2 = Math.max(...xs);
  box.y2 = Math.max(...ys);
  if (shapeType !== "detect") box.points = points;
}

function _refineBoxBadge(box) {
  return `${_refineClassDisplayName(box)} · ${_refineBoxCoords(box)}`;
}

function _refineSelectedBox() {
  return (state.refineImageData?.boxes || []).find((box) => box.box_id === state.refineSelectedBoxId) || null;
}

function _refineSetSelectedBox(boxId) {
  state.refineSelectedBoxId = boxId || "";
  renderRefineBoxList();
  renderRefineSelectedBox();
  renderRefineOverlay();
}

function _refineMarkDirty(dirty = true, note = "") {
  state.refineDirty = Boolean(dirty);
  _refineSetImageStatus(note);
}

function _refineEnsureDefaultClassValue() {
  const options = _refineClassOptions();
  const valid = options.some((item) => _refineEncodeClassOption(item) === state.refineDefaultClassValue);
  if (!valid) {
    state.refineDefaultClassValue = options[0] ? _refineEncodeClassOption(options[0]) : "";
  }
}

function _refineSetCreateMode(enabled) {
  const next = Boolean(enabled);
  if (next && !state.refineImageData) {
    toast("请先选择图片", "warning");
    _refineSetImageStatus();
    return false;
  }
  const activeShapeType = normalizeLabelTask($("refineShapeType")?.value || state.refineImageData?.label_task || "detect");
  if (!next && state.refineCreateMode && activeShapeType === "segment" && (state.refineDraftPoints || []).length) {
    if (!_refineCommitDraftShape()) {
      _refineSetImageStatus(`已添加 ${state.refineDraftPoints.length} 个点，继续点击或按 S 完成`);
      return false;
    }
  }
  if (state.refineCreateMode === next) {
    _refineSetImageStatus(next ? _refineCreateModeHint(activeShapeType) : "A/D 切图，W 开启新增");
    return false;
  }
  state.refineCreateMode = next;
  state.refineDraftPoints = [];
  state.refineDraftHoverPoint = null;
  renderRefineOverlay();
  _refineSetImageStatus(next ? _refineCreateModeHint(activeShapeType) : "A/D 切图，W 开启新增");
  return true;
}

function _refineStepDefaultClass(step) {
  const options = _refineClassOptions();
  if (!options.length) {
    toast("当前没有可用类别", "warning");
    return;
  }
  _refineEnsureDefaultClassValue();
  const values = options.map((item) => _refineEncodeClassOption(item));
  const currentIdx = Math.max(0, values.indexOf(state.refineDefaultClassValue));
  const nextIdx = (currentIdx + step + values.length) % values.length;
  const nextValue = values[nextIdx];
  const nextChoice = _refineDecodeClassOption(nextValue);
  if (!nextChoice) return;
  state.refineDefaultClassValue = nextValue;
  const note = `默认类别：${_refineClassDisplayName(nextChoice)}`;
  if (_refineSelectedBox()) {
    _refineApplyClassToSelected(nextValue, note);
    return;
  }
  renderRefineClassSelects();
  _refineSetImageStatus(note);
}

function renderRefineClassSelects() {
  const selectedClass = $("refineSelectedClass");
  const newBoxClass = $("refineNewBoxClass");
  const shapeTypeSel = $("refineShapeType");
  const options = _refineClassOptions();
  _refineEnsureDefaultClassValue();
  if (shapeTypeSel) {
    const lockedTask = _refineLockedShapeTask();
    const currentTask = normalizeLabelTask(shapeTypeSel.value || state.refineImageData?.label_task || "detect");
    shapeTypeSel.disabled = Boolean(lockedTask);
    shapeTypeSel.value = lockedTask || currentTask;
  }

  if (newBoxClass) {
    newBoxClass.innerHTML = options.map((item) => `<option value="${_esc(_refineEncodeClassOption(item))}">${_esc(item.name)}${item.class_id === null ? "（新）" : ` (#${item.class_id})`}</option>`).join("");
    if (state.refineDefaultClassValue) newBoxClass.value = state.refineDefaultClassValue;
  }
  if (selectedClass) {
    const box = _refineSelectedBox();
    selectedClass.innerHTML = options.map((item) => `<option value="${_esc(_refineEncodeClassOption(item))}">${_esc(item.name)}${item.class_id === null ? "（新）" : ` (#${item.class_id})`}</option>`).join("");
    if (box) {
      const desired = _refineEncodeClassOption({ class_id: box.class_id, name: box.class_name });
      if ([...selectedClass.options].some((opt) => opt.value === desired)) selectedClass.value = desired;
    }
  }
}

function renderRefineImageMeta() {
  const box = $("refineMeta");
  if (!box) return;
  const data = state.refineImageData;
  if (!data) {
    box.innerHTML = '<div class="muted-text">请选择图片集并加载图片</div>';
    return;
  }
  box.innerHTML = `
    <div><strong>${_esc(data.filename)}</strong></div>
    <div class="muted-text">标签类型 ${_esc(labelTaskLabel(data.label_task || "detect"))}</div>
    <div class="muted-text">尺寸 ${data.image_width} × ${data.image_height}</div>
    <div class="muted-text">审核 ${_esc(data.review_status || 'todo')} · ${_esc(data.reviewer || '未填写')}</div>
    <div class="muted-text">备注 ${_esc(data.review_note || '无')}</div>
    <div class="muted-text">回滚 ${data.can_rollback ? `可用（${_esc(data.last_backup_at || '最近一次')})` : '暂无快照'}</div>
  `;
}

function renderRefineImageSelect() {
  const sel = $("refineImageSelect");
  if (!sel) return;
  const old = state.refineCurrentImageId;
  sel.innerHTML = "";
  state.refineImages.forEach((item, idx) => {
    const opt = document.createElement("option");
    opt.value = item.image_id;
    opt.textContent = `${idx + 1}. ${item.filename}${item.label_exists ? ' [有标注]' : ' [无标注]'}`;
    sel.appendChild(opt);
  });
  if (old && state.refineImages.some((item) => item.image_id === old)) {
    sel.value = old;
  }
}

function renderRefineBoxList() {
  const box = $("refineBoxList");
  if (!box) return;
  const items = _refineSortBoxes(state.refineImageData?.boxes || []);
  if (!items.length) {
    box.innerHTML = '<div class="muted-text">当前图片暂无框</div>';
    return;
  }
  box.innerHTML = items.map((item, idx) => `
    <button type="button" class="refine-box-row ${item.box_id === state.refineSelectedBoxId ? 'active' : ''}" data-box-id="${_esc(item.box_id)}">
      <span class="refine-box-row-index">${item.class_id === null || item.class_id === undefined ? "" : _esc(String(item.class_id))}</span>
      <span class="refine-box-row-label">${_esc(`${String(item.class_name || "").trim() || "未命名"} · ${labelTaskLabel(item.shape_type || "detect")} · ${_refineBoxCoords(item)}`)}</span>
    </button>
  `).join("");
  box.querySelectorAll(".refine-box-row").forEach((el) => {
    el.addEventListener("click", () => {
      _refineSetSelectedBox(el.dataset.boxId || "");
      _refineFocusWorkspace();
    });
  });
}

function renderRefineSelectedBox() {
  const empty = $("refineSelectedEmpty");
  const panel = $("refineSelectedPanel");
  const coords = $("refineSelectedCoords");
  const box = _refineSelectedBox();
  if (!empty || !panel || !coords) return;
  if (!box) {
    empty.style.display = "";
    panel.style.display = "none";
    return;
  }
  empty.style.display = "none";
  panel.style.display = "";
  renderRefineClassSelects();
  coords.textContent = `${labelTaskLabel(box.shape_type || "detect")} · x1=${box.x1.toFixed(1)} y1=${box.y1.toFixed(1)} x2=${box.x2.toFixed(1)} y2=${box.y2.toFixed(1)}`;
}

function _syncOverlayToImage() {
  // stage 是 flex 居中容器，img 按原图比例缩放，小图会在 stage 里居中留空白
  // 把 overlay 动态定位到 img 的实际 offset+size，这样 box 用 % 就能天然贴合 img
  const stage = $("refineStage");
  const overlay = $("refineOverlay");
  const img = $("refineImage");
  if (!stage || !overlay || !img) return false;
  const stageRect = stage.getBoundingClientRect();
  const imgRect = img.getBoundingClientRect();
  if (!imgRect.width || !imgRect.height) return false;
  overlay.style.left = (imgRect.left - stageRect.left) + "px";
  overlay.style.top = (imgRect.top - stageRect.top) + "px";
  overlay.style.width = imgRect.width + "px";
  overlay.style.height = imgRect.height + "px";
  overlay.style.right = "auto";
  overlay.style.bottom = "auto";
  return true;
}

function _refineStageMetrics() {
  // overlay 已通过 _syncOverlayToImage 对齐到 img，此处直接用 overlay 的 rect 即为 img 的 rect
  const overlay = $("refineOverlay");
  const data = state.refineImageData;
  if (!overlay || !data) return null;
  const rect = overlay.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  return {
    rect,
    width: Number(data.image_width || 1),
    height: Number(data.image_height || 1),
  };
}

function _refinePointerToImage(ev) {
  const metrics = _refineStageMetrics();
  if (!metrics) return null;
  const px = ((ev.clientX - metrics.rect.left) / metrics.rect.width) * metrics.width;
  const py = ((ev.clientY - metrics.rect.top) / metrics.rect.height) * metrics.height;
  return {
    x: Math.min(Math.max(px, 0), metrics.width),
    y: Math.min(Math.max(py, 0), metrics.height),
  };
}

function _refinePointerInsideImage(ev) {
  const metrics = _refineStageMetrics();
  if (!metrics) return false;
  const r = metrics.rect;
  return ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
}

function _refineClampBox(box) {
  const width = Number(state.refineImageData?.image_width || 1);
  const height = Number(state.refineImageData?.image_height || 1);
  if (box.shape_type && box.shape_type !== "detect" && Array.isArray(box.points)) {
    box.points = box.points.map((point) => [
      Math.min(Math.max(Number(point?.[0] || 0), 0), width),
      Math.min(Math.max(Number(point?.[1] || 0), 0), height),
    ]);
    _refineSyncBoxBounds(box);
    return;
  }
  let x1 = Math.min(Number(box.x1 || 0), Number(box.x2 || 0));
  let x2 = Math.max(Number(box.x1 || 0), Number(box.x2 || 0));
  let y1 = Math.min(Number(box.y1 || 0), Number(box.y2 || 0));
  let y2 = Math.max(Number(box.y1 || 0), Number(box.y2 || 0));
  x1 = Math.min(Math.max(0, x1), width);
  x2 = Math.min(Math.max(0, x2), width);
  y1 = Math.min(Math.max(0, y1), height);
  y2 = Math.min(Math.max(0, y2), height);
  box.x1 = x1;
  box.x2 = x2;
  box.y1 = y1;
  box.y2 = y2;
}

function _refineCreateBoxId() {
  return `box_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
}

function _refineCommitDraftShape() {
  if (!state.refineImageData || !state.refineCreateMode) return false;
  const shapeType = normalizeLabelTask($("refineShapeType")?.value || state.refineImageData.label_task || "detect");
  if (shapeType === "detect") return false;
  const minPoints = _refineMinDraftPoints(shapeType);
  if ((state.refineDraftPoints || []).length < minPoints) {
    toast(`至少需要 ${minPoints} 个点`, "warning");
    return false;
  }
  const choice = _refineDecodeClassOption(state.refineDefaultClassValue || $("refineNewBoxClass")?.value || "");
  if (!choice) return false;
  const points = shapeType === "obb"
    ? _refineNormalizeObbPoints(state.refineDraftPoints.slice(0, 4))
    : state.refineDraftPoints.slice();
  const box = {
    box_id: _refineCreateBoxId(),
    class_id: choice.class_id,
    class_name: choice.name,
    shape_type: shapeType,
    points,
    x1: 0, y1: 0, x2: 0, y2: 0,
  };
  _refineSyncBoxBounds(box);
  state.refineImageData.boxes.push(box);
  state.refineImageData.boxes = _refineSortBoxes(state.refineImageData.boxes);
  state.refineDraftPoints = [];
  state.refineDraftHoverPoint = null;
  _refineSetSelectedBox(box.box_id);
  renderRefineOverlay();
  _refineMarkDirty(true, "已新增形状");
  return true;
}

function renderRefineOverlay() {
  const img = $("refineImage");
  const overlay = $("refineOverlay");
  const empty = $("refineCanvasEmpty");
  const button = $("btnRefineCreateBox");
  if (!img || !overlay || !empty || !button) return;
  const data = state.refineImageData;
  if (!data || !data.image_url) {
    overlay.innerHTML = "";
    img.removeAttribute("src");
    empty.style.display = "";
    button.textContent = "新增框";
    $("refineStage")?.classList.remove("create-mode");
    return;
  }
  if (img.dataset.rawSrc !== data.image_url) {
    img.dataset.rawSrc = data.image_url;
    img.src = data.image_url;
  }
  empty.style.display = "none";
  overlay.innerHTML = "";
  const selectedBoxId = state.refineSelectedBoxId || "";
  const editingBoxId = state.refineInteraction?.box_id || "";
  overlay.classList.toggle("has-selection", Boolean(selectedBoxId));
  // 先把 overlay 对齐到 img 实际位置；img 还没 load 时跳过这次绘制，等 img load 事件会再触发
  if (!_syncOverlayToImage()) return;
  const width = Number(data.image_width || 1);
  const height = Number(data.image_height || 1);
  (data.boxes || []).forEach((box) => {
    _refineSyncBoxBounds(box);
    const el = document.createElement("div");
    const shapeType = normalizeLabelTask(box.shape_type || data.label_task || "detect");
    const color = _refineClassColor(box);
    const isSelected = box.box_id === selectedBoxId;
    const isEditing = box.box_id === editingBoxId;
    el.className = `refine-box shape-${shapeType} ${isSelected ? "selected" : ""} ${isEditing ? "editing" : ""}`.trim();
    el.dataset.boxId = box.box_id;
    el.style.setProperty("--refine-color", color.stroke);
    el.style.setProperty("--refine-fill", color.fill);
    el.style.setProperty("--refine-label-bg", color.label);
    el.style.zIndex = isEditing ? "40" : (isSelected ? "30" : "10");
    el.title = _refineBoxBadge(box);
    const labelInside = box.y1 <= height * 0.06;
    // overlay 已贴合 img，box 用 % 即为图片坐标系
    el.style.left = `${(box.x1 / width) * 100}%`;
    el.style.top = `${(box.y1 / height) * 100}%`;
    el.style.width = `${((box.x2 - box.x1) / width) * 100}%`;
    el.style.height = `${((box.y2 - box.y1) / height) * 100}%`;
    const shapePoints = _refineShapePoints(box);
    const localPoints = shapePoints.map((point) => {
      const bx = Math.max(1, box.x2 - box.x1);
      const by = Math.max(1, box.y2 - box.y1);
      return `${((point[0] - box.x1) / bx) * 100},${((point[1] - box.y1) / by) * 100}`;
    }).join(" ");
    el.innerHTML = `
      <span class="refine-box-label${labelInside ? " inside" : ""}">${_esc(_refineClassDisplayName(box))}</span>
      ${shapeType === "detect" ? `
        <span class="refine-handle nw" data-handle="nw"></span>
        <span class="refine-handle ne" data-handle="ne"></span>
        <span class="refine-handle sw" data-handle="sw"></span>
        <span class="refine-handle se" data-handle="se"></span>
      ` : `
        <svg class="refine-shape-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon class="refine-shape-poly" points="${localPoints}"></polygon>
        </svg>
      `}
    `;
    if (shapeType !== "detect") {
      const bx = Math.max(1, box.x2 - box.x1);
      const by = Math.max(1, box.y2 - box.y1);
      shapePoints.forEach((point, pointIndex) => {
        const handle = document.createElement("span");
        handle.className = "refine-point-handle";
        handle.dataset.pointIndex = String(pointIndex);
        handle.style.left = `${((point[0] - box.x1) / bx) * 100}%`;
        handle.style.top = `${((point[1] - box.y1) / by) * 100}%`;
        el.appendChild(handle);
      });
      if (shapeType === "obb") {
        const metrics = _refineObbMetrics(shapePoints);
        if (metrics) {
          const rotateDistance = Math.max(24, Math.min(48, metrics.height * 0.35));
          const rotatePoint = [
            metrics.topMid[0] + (metrics.topUnit[0] * rotateDistance),
            metrics.topMid[1] + (metrics.topUnit[1] * rotateDistance),
          ];
          const line = document.createElement("span");
          line.className = "refine-rotate-line";
          line.style.left = `${((metrics.topMid[0] - box.x1) / bx) * 100}%`;
          line.style.top = `${((metrics.topMid[1] - box.y1) / by) * 100}%`;
          line.style.width = `${(rotateDistance / bx) * 100}%`;
          line.style.transform = `rotate(${Math.atan2(rotatePoint[1] - metrics.topMid[1], rotatePoint[0] - metrics.topMid[0])}rad)`;
          el.appendChild(line);

          const rotateHandle = document.createElement("span");
          rotateHandle.className = "refine-rotate-handle";
          rotateHandle.dataset.rotateHandle = "1";
          rotateHandle.style.left = `${((rotatePoint[0] - box.x1) / bx) * 100}%`;
          rotateHandle.style.top = `${((rotatePoint[1] - box.y1) / by) * 100}%`;
          el.appendChild(rotateHandle);
        }
      }
    }
    el.addEventListener("mousedown", function(ev) {
      if (ev.button !== 0) return;
      // 正在创建/拖拽中，不覆盖当前 interaction
      if (state.refineInteraction) return;
      if (state.refineCreateMode) return;
      ev.preventDefault();
      ev.stopPropagation();
      state.refineSelectedBoxId = box.box_id;
      var handle = ev.target?.dataset?.handle || "";
      var pointIndex = ev.target?.dataset?.pointIndex;
      var rotateHandle = ev.target?.dataset?.rotateHandle === "1";
      var point = _refinePointerToImage(ev);
      if (!point) return;
      state.refineInteraction = {
        type: rotateHandle
          ? "rotate-obb"
          : (pointIndex !== undefined
            ? (shapeType === "obb" ? "resize-obb-corner" : "point")
            : (shapeType === "detect" ? (handle ? "resize" : "move") : "move-shape")),
        handle: handle,
        pointIndex: pointIndex !== undefined ? Number(pointIndex) : -1,
        box_id: box.box_id,
        start: point,
        original: { ...box, points: shapePoints.map((p) => [p[0], p[1]]) },
        moved: false,
      };
      renderRefineBoxList();
      renderRefineSelectedBox();
      renderRefineOverlay();
    });
    overlay.appendChild(el);
  });
  if (state.refineCreateMode && state.refineDraftPoints.length >= 1 && normalizeLabelTask($("refineShapeType")?.value || data.label_task || "detect") === "segment") {
    const svg = document.createElement("svg");
    const draftShapeType = normalizeLabelTask($("refineShapeType")?.value || data.label_task || "detect");
    const hoverPoint = state.refineDraftHoverPoint ? [state.refineDraftHoverPoint.x, state.refineDraftHoverPoint.y] : null;
    const draftPoints = state.refineDraftPoints.slice();
    const previewPoints = [...draftPoints, ...(hoverPoint ? [hoverPoint] : [])];
    const previewShape = draftShapeType === "segment" && previewPoints.length >= 3;
    const pathMarkup = previewShape
      ? `<polygon points="${previewPoints.map((p) => `${p[0]},${p[1]}`).join(" ")}" fill="rgba(16,185,129,0.18)" stroke="rgba(16,185,129,0.98)" stroke-width="2.5"></polygon>`
      : `<polyline points="${previewPoints.map((p) => `${p[0]},${p[1]}`).join(" ")}" fill="none" stroke="rgba(16,185,129,0.98)" stroke-width="2.5"></polyline>`;
    const pointMarkup = draftPoints.map((p) => (
      `<circle cx="${p[0]}" cy="${p[1]}" r="6" fill="rgba(16,185,129,0.98)" stroke="rgba(255,255,255,0.95)" stroke-width="2"></circle>`
    )).join("") + (hoverPoint
      ? `<circle cx="${hoverPoint[0]}" cy="${hoverPoint[1]}" r="4.5" fill="rgba(16,185,129,0.72)" stroke="rgba(255,255,255,0.9)" stroke-width="1.5"></circle>`
      : "");
    svg.className = "refine-draft-poly";
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `${pathMarkup}${pointMarkup}`;
    overlay.appendChild(svg);
  }
  button.textContent = state.refineCreateMode ? "结束新增" : "新增框";
  $("refineStage")?.classList.toggle("create-mode", state.refineCreateMode);
}

function _refineApplyClassToSelected(value, note = "类别已修改") {
  const box = _refineSelectedBox();
  const choice = _refineDecodeClassOption(value);
  if (!box || !choice) return false;
  box.class_id = choice.class_id;
  box.class_name = choice.name;
  if (choice.class_id === null) {
    const exists = state.refinePendingClassNames.some((item) => item.toLowerCase() === choice.name.toLowerCase());
    if (!exists) state.refinePendingClassNames.push(choice.name);
  }
  state.refineImageData.boxes = _refineSortBoxes(state.refineImageData.boxes || []);
  renderRefineClassSelects();
  renderRefineBoxList();
  renderRefineOverlay();
  renderRefineSelectedBox();
  _refineMarkDirty(true, note);
  return true;
}

function _refineBeginCreate(ev) {
  const choice = _refineDecodeClassOption(state.refineDefaultClassValue || $("refineNewBoxClass")?.value || "");
  if (!choice) {
    toast("请先选择新框默认类别", "warning");
    return;
  }
  // 小图在画布里有留边，只有点在图片内才允许创建新框
  if (!_refinePointerInsideImage(ev)) return;
  const point = _refinePointerToImage(ev);
  if (!point || !state.refineImageData) return;
  const shapeType = normalizeLabelTask($("refineShapeType")?.value || state.refineImageData.label_task || "detect");
  if (shapeType === "segment") {
    state.refineDraftPoints.push([point.x, point.y]);
    state.refineDraftHoverPoint = point;
    renderRefineOverlay();
    _refineSetImageStatus(`SEG 已添加 ${state.refineDraftPoints.length} 个点，继续移动预览，按 S 完成多边形`);
    return;
  }
  if (shapeType === "obb") {
    const points = _refineRectPointsFromDiagonal([point.x, point.y], [point.x + 1, point.y + 1]);
    const box = {
      box_id: _refineCreateBoxId(),
      class_id: choice.class_id,
      class_name: choice.name,
      shape_type: "obb",
      points,
      x1: point.x,
      y1: point.y,
      x2: point.x + 1,
      y2: point.y + 1,
    };
    _refineSyncBoxBounds(box);
    state.refineImageData.boxes.push(box);
    state.refineImageData.boxes = _refineSortBoxes(state.refineImageData.boxes);
    _refineSetSelectedBox(box.box_id);
    state.refineInteraction = {
      type: "create-obb",
      handle: "",
      box_id: box.box_id,
      start: point,
      original: { ...box, points: points.map((p) => [p[0], p[1]]) },
      moved: false,
    };
    renderRefineOverlay();
    _refineSetImageStatus("OBB 拖拽中：松开鼠标完成初始框");
    return;
  }
  const box = {
    box_id: _refineCreateBoxId(),
    class_id: choice.class_id,
    class_name: choice.name,
    shape_type: "detect",
    points: [],
    x1: point.x,
    y1: point.y,
    x2: point.x + 1,
    y2: point.y + 1,
  };
  state.refineImageData.boxes.push(box);
  state.refineImageData.boxes = _refineSortBoxes(state.refineImageData.boxes);
  _refineSetSelectedBox(box.box_id);
  state.refineInteraction = {
    type: "create",
    handle: "",
    box_id: box.box_id,
    start: point,
    original: { ...box },
    moved: false,
  };
  renderRefineOverlay();
}

function _refineDeleteSelectedBox() {
  const items = state.refineImageData?.boxes || [];
  const next = items.filter((item) => item.box_id !== state.refineSelectedBoxId);
  if (next.length === items.length) return;
  state.refineImageData.boxes = _refineSortBoxes(next);
  state.refineSelectedBoxId = next[0]?.box_id || "";
  renderRefineBoxList();
  renderRefineSelectedBox();
  renderRefineOverlay();
  _refineMarkDirty(true, "已删除选中框");
}

function _refineHandlePointerMove(ev) {
  if (!state.refineInteraction && state.refineCreateMode && state.refineImageData) {
    const shapeType = normalizeLabelTask($("refineShapeType")?.value || state.refineImageData.label_task || "detect");
    if (shapeType === "segment") {
      if (!_refinePointerInsideImage(ev)) {
        if (state.refineDraftHoverPoint) {
          state.refineDraftHoverPoint = null;
          renderRefineOverlay();
        }
        return;
      }
      const hoverPoint = _refinePointerToImage(ev);
      if (hoverPoint && !_refineSamePoint(state.refineDraftHoverPoint, hoverPoint)) {
        state.refineDraftHoverPoint = hoverPoint;
        renderRefineOverlay();
      }
      return;
    }
    if (state.refineDraftHoverPoint) {
      state.refineDraftHoverPoint = null;
      renderRefineOverlay();
    }
  }
  const interaction = state.refineInteraction;
  if (!interaction || !state.refineImageData) return;
  const point = _refinePointerToImage(ev);
  if (!point) return;
  const box = (state.refineImageData.boxes || []).find((item) => item.box_id === interaction.box_id);
  if (!box) return;
  const original = interaction.original;
  const handle = interaction.handle || "";
  const deltaX = point.x - interaction.start.x;
  const deltaY = point.y - interaction.start.y;
  if (!interaction.moved && Math.hypot(deltaX, deltaY) >= 0.5) {
    interaction.moved = true;
  }
  if (interaction.type === "point") {
    const pts = _refineShapePoints(box);
    pts[interaction.pointIndex] = [point.x, point.y];
    box.points = normalizeLabelTask(box.shape_type || "detect") === "obb"
      ? _refineNormalizeObbPoints(pts)
      : pts;
  } else if (interaction.type === "resize-obb-corner") {
    box.points = _refineResizeObbFromCorner(original.points || _refineShapePoints(box), interaction.pointIndex, point);
  } else if (interaction.type === "rotate-obb") {
    box.points = _refineRotateObbByPointer(original.points || _refineShapePoints(box), point);
  } else if (interaction.type === "move-shape") {
    const dx = point.x - interaction.start.x;
    const dy = point.y - interaction.start.y;
    box.points = (original.points || []).map((p) => [p[0] + dx, p[1] + dy]);
  } else if (interaction.type === "move") {
    const dx = point.x - interaction.start.x;
    const dy = point.y - interaction.start.y;
    const width = original.x2 - original.x1;
    const height = original.y2 - original.y1;
    box.x1 = original.x1 + dx;
    box.y1 = original.y1 + dy;
    box.x2 = box.x1 + width;
    box.y2 = box.y1 + height;
  } else if (interaction.type === "create-obb") {
    box.points = _refineRectPointsFromDiagonal([interaction.start.x, interaction.start.y], [point.x, point.y]);
    _refineSyncBoxBounds(box);
  } else {
    box.x1 = original.x1;
    box.y1 = original.y1;
    box.x2 = original.x2;
    box.y2 = original.y2;
    if (interaction.type === "create" || handle.includes("e")) box.x2 = point.x;
    if (interaction.type === "create" || handle.includes("s")) box.y2 = point.y;
    if (handle.includes("w")) box.x1 = point.x;
    if (handle.includes("n")) box.y1 = point.y;
  }
  _refineClampBox(box);
  renderRefineOverlay();
  renderRefineSelectedBox();
}

function _refineHandlePointerUp() {
  const interaction = state.refineInteraction;
  if (!interaction || !state.refineImageData) return;
  const box = (state.refineImageData.boxes || []).find((item) => item.box_id === interaction.box_id);
  state.refineInteraction = null;
  if (!box) return;
  if (!interaction.moved && interaction.type !== "create" && interaction.type !== "create-obb") {
    renderRefineBoxList();
    renderRefineSelectedBox();
    renderRefineOverlay();
    _refineSetImageStatus();
    return;
  }
  const width = Math.abs(box.x2 - box.x1);
  const height = Math.abs(box.y2 - box.y1);
  if (width < 2 || height < 2) {
    state.refineImageData.boxes = (state.refineImageData.boxes || []).filter((item) => item.box_id !== box.box_id);
    state.refineSelectedBoxId = state.refineImageData.boxes[0]?.box_id || "";
    renderRefineBoxList();
    renderRefineSelectedBox();
    renderRefineOverlay();
    toast("框过小，已取消本次编辑", "warning");
    _refineSetImageStatus();
    return;
  }
  renderRefineBoxList();
  renderRefineSelectedBox();
  renderRefineOverlay();
  const doneNote = interaction.type === "create-obb"
    ? "已新增 OBB"
    : interaction.type === "create"
      ? "已新增框"
      : "框已更新";
  _refineMarkDirty(true, doneNote);
}

async function loadRefineImage(imageId, options = {}) {
  const { skipDirty = false, quiet = false } = options;
  if (!imageId) {
    state.refineImageData = null;
    state.refineCurrentImageId = "";
    state.refineSelectedBoxId = "";
    state.refineDirty = false;
    state.refinePendingClassNames = [];
    state.refineDraftPoints = [];
    state.refineDraftHoverPoint = null;
    state.refineCreateMode = false;
    state.refineInteraction = null;
    renderRefineImageMeta();
    renderRefineClassSelects();
    renderRefineBoxList();
    renderRefineSelectedBox();
    renderRefineOverlay();
    _refineStatus("当前图片集暂无图片");
    _refineSetImageStatus();
    return true;
  }
  if (!skipDirty && !_refineConfirmDiscard()) return false;
  const data = await api(`/api/images/${imageId}/refine`);
  state.refineImageData = _refineNormalizePayload(data);
  state.refineCurrentImageId = imageId;
  state.refineSelectedBoxId = state.refineImageData.boxes[0]?.box_id || "";
  state.refinePendingClassNames = [];
  state.refineCreateMode = false;
  state.refineDraftPoints = [];
  state.refineDraftHoverPoint = null;
  state.refineInteraction = null;
  state.refineDirty = false;
  renderRefineImageMeta();
  renderRefineClassSelects();
  renderRefineBoxList();
  renderRefineSelectedBox();
  renderRefineOverlay();
  if (!quiet) _refineStatus(`已加载 ${state.refineImageData.filename}`);
  _refineSetImageStatus();
  const sel = $("refineImageSelect");
  if (sel) sel.value = imageId;
  _refineFocusWorkspace();
  return true;
}

async function loadRefineImageset(options = {}) {
  const { preserveImageId = "", skipDirty = false, quiet = false } = options;
  const imagesetId = $("refineImageset")?.value || "";
  if (!imagesetId) {
    state.refineImages = [];
    renderRefineImageSelect();
    await loadRefineImage("", { skipDirty: true, quiet: true });
    _refineStatus("请选择图片集");
    return;
  }
  if (!skipDirty && !_refineConfirmDiscard()) return;
  const data = await api(`/api/imagesets/${imagesetId}/images?page=1&page_size=2000`);
  state.refineImages = data.items || [];
  renderRefineImageSelect();
  const shapeTypeSel = $("refineShapeType");
  const imagesetMeta = state.imagesetMeta?.[imagesetId];
  if (shapeTypeSel && imagesetMeta?.label_count) {
    shapeTypeSel.value = normalizeLabelTask(imagesetMeta.label_task || "detect");
  }
  if (!state.refineImages.length) {
    await loadRefineImage("", { skipDirty: true, quiet: true });
    _refineStatus("当前图片集暂无图片");
    return;
  }
  const targetId = state.refineImages.some((item) => item.image_id === preserveImageId)
    ? preserveImageId
    : state.refineImages.some((item) => item.image_id === state.refineCurrentImageId)
      ? state.refineCurrentImageId
      : state.refineImages[0].image_id;
  const ok = await loadRefineImage(targetId, { skipDirty: true, quiet: true });
  if (ok && !quiet) {
    _refineStatus(`已加载 ${state.refineImages.length} 张图片，可逐张精修`);
  }
  _refineSetImageStatus("A/D 切图，W 开启新增，Ctrl/Cmd+S 保存");
  if (ok) _refineFocusWorkspace();
}

function _refineRefreshDependentViews(imagesetId) {
  if (!imagesetId) return;
  if ($("annotateImageset")?.value === imagesetId) loadInlinePreview(imagesetId, "annotateInlinePreview");
  if ($("aiImagesetSelect")?.value === imagesetId) loadInlinePreview(imagesetId, "aiInlinePreview");
  if ($("trainImageset")?.value === imagesetId) {
    updateTrainDatasetInfo();
    loadInlinePreview(imagesetId, "trainInlinePreview", { collapsed: true });
  }
  if (_previewModal.imagesetId === imagesetId && !$("imagesetPreviewModal")?.classList.contains("hidden")) {
    loadPreviewModal();
  }
}

async function saveRefineCurrentImage() {
  const data = state.refineImageData;
  if (!data) { toast("请先选择图片", "warning"); return; }
  const sortedBoxes = _refineSortBoxes(data.boxes || []);
  const payload = {
    boxes: sortedBoxes.map((box) => ({
      box_id: box.box_id,
      class_id: box.class_id,
      class_name: box.class_name,
      shape_type: normalizeLabelTask(box.shape_type || data.label_task || "detect"),
      points: normalizeLabelTask(box.shape_type || data.label_task || "detect") === "detect" ? [] : _refineShapePoints(box),
      x1: box.x1,
      y1: box.y1,
      x2: box.x2,
      y2: box.y2,
    })),
    new_classes: _refineClassOptions().filter((item) => item.class_id === null).map((item) => ({ name: item.name })),
    operator: ($("refineOperator")?.value || "anonymous").trim() || "anonymous",
  };
  try {
    _refineSetImageStatus("保存中...");
    await api(`/api/images/${data.image_id}/refine`, { method: "POST", body: payload });
    await refreshImagesetSelects();
    const imagesetId = $("refineImageset")?.value || data.imageset_id;
    if ($("refineImageset")) $("refineImageset").value = imagesetId;
    await loadRefineImageset({ preserveImageId: data.image_id, skipDirty: true, quiet: true });
    _refineRefreshDependentViews(imagesetId);
    state.refineDirty = false;
    _refineStatus(`已保存 ${data.filename}`);
    _refineSetImageStatus("保存完成");
    toast("当前图片已保存", "success");
  } catch (e) {
    _refineSetImageStatus(`保存失败: ${e.message}`);
    toast(e.message, "error");
  }
}

async function rollbackRefineCurrentImage() {
  const data = state.refineImageData;
  if (!data) { toast("请先选择图片", "warning"); return; }
  if (!window.confirm("确定回滚当前图片到上一次保存前的状态吗？")) return;
  try {
    _refineSetImageStatus("回滚中...");
    await api(`/api/images/${data.image_id}/refine/rollback`, { method: "POST" });
    await refreshImagesetSelects();
    const imagesetId = $("refineImageset")?.value || data.imageset_id;
    if ($("refineImageset")) $("refineImageset").value = imagesetId;
    await loadRefineImageset({ preserveImageId: data.image_id, skipDirty: true, quiet: true });
    _refineRefreshDependentViews(imagesetId);
    state.refineDirty = false;
    _refineStatus(`已回滚 ${data.filename}`);
    _refineSetImageStatus("回滚完成");
    toast("当前图片已回滚", "success");
  } catch (e) {
    _refineSetImageStatus(`回滚失败: ${e.message}`);
    toast(e.message, "error");
  }
}

async function _refineMoveWithAutoSave(step) {
  // 方向键切图前，若当前有未保存修改就先自动保存；保存失败则不切，避免丢数据
  if (!state.refineImages.length) return;
  const idx = state.refineImages.findIndex(function(item) { return item.image_id === state.refineCurrentImageId; });
  const next = state.refineImages[idx + step];
  if (!next) return;
  if (state.refineDirty && state.refineImageData) {
    try {
      await saveRefineCurrentImage();
    } catch (err) {
      toast("自动保存失败，未切换图片: " + (err?.message || err), "error");
      return;
    }
    if (state.refineDirty) return; // 保存未成功（内部已 toast）
  }
  try {
    const ok = await loadRefineImage(next.image_id, { skipDirty: true });
    if (!ok && $("refineImageSelect")) $("refineImageSelect").value = state.refineCurrentImageId;
  } catch (err) {
    toast(err?.message || String(err), "error");
  }
}

function _refineMoveImage(step) {
  if (!state.refineImages.length) return;
  const idx = state.refineImages.findIndex((item) => item.image_id === state.refineCurrentImageId);
  const next = state.refineImages[idx + step];
  if (!next) return;
  loadRefineImage(next.image_id).then((ok) => {
    if (!ok && $("refineImageSelect")) $("refineImageSelect").value = state.refineCurrentImageId;
  }).catch((e) => toast(e.message, "error"));
}

function _refineCreateOrAssignClass() {
  const raw = prompt("请输入新类别名称:");
  const name = String(raw || "").trim();
  if (!name) return;
  const existing = _refineClassOptions().find((item) => item.name.toLowerCase() === name.toLowerCase());
  const value = existing ? _refineEncodeClassOption(existing) : `${REFINE_NEW_CLASS_PREFIX}${encodeURIComponent(name)}`;
  if (!existing) {
    const seen = state.refinePendingClassNames.some((item) => item.toLowerCase() === name.toLowerCase());
    if (!seen) state.refinePendingClassNames.push(name);
  }
  const selected = _refineSelectedBox();
  if (selected) {
    _refineApplyClassToSelected(value);
  } else {
    state.refineDefaultClassValue = value;
    renderRefineClassSelects();
    _refineStatus(`已准备新类别：${name}`);
  }
}
