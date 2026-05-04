const CLASS_PRESET_STORAGE_KEY = "autoannotation.classMappingPresets.v1";

function computeCurrentMappingSignature() {
  const modelId = $("modelSelect")?.value || "";
  const selected = getSelectedClassIds()
    .map((x) => Number(x))
    .filter((x) => Number.isInteger(x))
    .sort((a, b) => a - b);
  const overrides = Object.entries(getClassIdOverrides() || {})
    .map(([src, dst]) => [Number(src), Number(dst)])
    .filter(([src, dst]) => Number.isInteger(src) && Number.isInteger(dst))
    .sort((a, b) => a[0] - b[0]);
  return JSON.stringify({
    modelId,
    targetClasses: [...state.targetClasses],
    selectedClassIds: selected,
    overrides,
  });
}

function setMappingConfirmed(confirmed, text = "") {
  state.mappingConfirmed = Boolean(confirmed);
  if (!state.mappingConfirmed) {
    state.mappingSignature = "";
  }
  const badge = $("mappingConfirmBadge");
  if (badge) {
    badge.textContent = state.mappingConfirmed ? "已确认" : "未确认";
    badge.style.color = state.mappingConfirmed ? "var(--ok)" : "var(--ink-muted)";
    badge.style.fontWeight = "700";
  }
  const status = $("mappingConfirmStatus");
  if (status) {
    status.textContent = text || (
      state.mappingConfirmed
        ? "映射已确认：当前配置可启动权重自动打标。"
        : "未确认映射：请检查来源类别与目标ID映射后，点击\"确认映射（必做）\"。"
    );
  }
  const startBtn = $("btnStartAnnotate");
  if (startBtn) {
    startBtn.disabled = !state.mappingConfirmed;
    startBtn.title = state.mappingConfirmed ? "开始权重自动打标" : "请先确认映射";
  }
}

/* ========== Class Presets ========== */
function readClassPresetsFromStorage() {
  try {
    const raw = localStorage.getItem(CLASS_PRESET_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((it) => it && typeof it === "object") : [];
  } catch (_) { return []; }
}

function writeClassPresetsToStorage(items) {
  try { localStorage.setItem(CLASS_PRESET_STORAGE_KEY, JSON.stringify(items || [])); } catch (_) {}
}

function sourceClassSignature(classes) {
  return (classes || []).map((x) => `${x.id}:${x.name}`).sort().join("|");
}

function refreshClassPresetSelect() {
  const sel = $("classPresetSelect");
  if (!sel) return;
  const old = sel.value;
  sel.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = state.classPresets.length ? "请选择预设" : "暂无预设";
  sel.appendChild(empty);
  state.classPresets.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.preset_id;
    const modelMark = item.model_name ? ` / ${item.model_name}` : "";
    opt.textContent = `${item.name}${modelMark}`;
    sel.appendChild(opt);
  });
  if (old) sel.value = old;
}

function loadClassPresets() {
  state.classPresets = readClassPresetsFromStorage();
  refreshClassPresetSelect();
}

/* ========== Target Classes ========== */
function updateTargetClassesStatus() {
  const box = $("targetClassesStatus");
  if (!box) return;
  if (!state.targetClasses.length) {
    box.textContent = "未加载新场景类别，默认使用模型原始类别。";
    return;
  }
  const head = state.targetClasses.slice(0, 10).map((name, idx) => `${idx}:${name}`).join(", ");
  box.textContent = `已加载 ${state.targetClasses.length} 个：${head}` + (state.targetClasses.length > 10 ? " ..." : "");
}

function _classUiSnapshot() {
  const snap = {};
  Array.from($("classChecklist")?.querySelectorAll(".check-item") || []).forEach((row) => {
    const cb = row.querySelector("input[type=checkbox]");
    const sel = row.querySelector("select.target-id-map");
    if (!cb || !sel) return;
    const src = Number(cb.value);
    if (!Number.isInteger(src)) return;
    snap[src] = { checked: cb.checked, targetId: Number(sel.value) };
  });
  return snap;
}

function _defaultTargetIdForSource(srcId, srcName) {
  if (!state.targetClasses.length) return srcId;
  const srcLower = String(srcName || "").trim().toLowerCase();
  const exactIdx = state.targetClasses.findIndex((x) => String(x).trim().toLowerCase() === srcLower);
  if (exactIdx >= 0) return exactIdx;
  if (srcId >= 0 && srcId < state.targetClasses.length) return srcId;
  return 0;
}

function _syncCheckItemDisabled(row) {
  const cb = row.querySelector("input[type=checkbox]");
  const sel = row.querySelector("select.target-id-map");
  const arrow = row.querySelector(".map-arrow");
  const nameSpan = row.querySelector(".source-class");
  if (!cb || !sel) return;
  const off = !cb.checked;
  sel.disabled = off;
  sel.style.opacity = off ? "0.35" : "";
  if (arrow) arrow.style.opacity = off ? "0.35" : "";
  if (nameSpan) nameSpan.style.opacity = off ? "0.5" : "";
}

async function renderClassChecklist(previous = null) {
  const box = $("classChecklist");
  if (!box) return;
  box.innerHTML = "";
  const snap = previous || {};
  const imagesetId = $("annotateImageset")?.value || "";
  const modelId = $("modelSelect")?.value || "";
  const labelMode = $("labelMode")?.value || "append";

  // 从后端获取 mapClasses — single source of truth
  let mapClasses = [];
  if (modelId && state.sourceClasses.length) {
    try {
      const payload = {
        model_id: modelId,
        imageset_id: imagesetId,
        selected_class_ids: state.sourceClasses.map(function(x) { return x.id; }),
        target_classes: [...state.targetClasses],
        label_mode: labelMode,
      };
      const resp = await fetch("/api/annotate/resolve-mapping", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        const data = await resp.json();
        mapClasses = (data.items || []).map(function(item) { return { id: item.id, name: item.name }; });
      }
    } catch (err) {
      console.error("resolve-mapping 请求失败:", err);
    }
  }
  // fallback：后端无响应时用 sourceClasses 自身做 1:1 映射
  if (!mapClasses.length && state.sourceClasses.length) {
    mapClasses = state.sourceClasses.map(function(x) { return { id: x.id, name: x.name }; });
  }

  state.sourceClasses.forEach(function(cls) {
    const row = document.createElement("label");
    row.className = "check-item";
    const saved = snap[cls.id] || {};
    const checked = saved.checked !== undefined ? Boolean(saved.checked) : true;
    let targetId;
    if (Number.isInteger(saved.targetId)) {
      targetId = saved.targetId;
    } else {
      // Smart default: find the mapClasses entry matching this source class
      const srcName = cls.name.toLowerCase();
      const match = mapClasses.find(m => m.name.toLowerCase().replace(" (复用)", "") === srcName);
      targetId = match ? match.id : (mapClasses.length ? mapClasses[0].id : 0);
    }
    if (!mapClasses.some((x) => x.id === targetId)) {
      targetId = mapClasses.length ? mapClasses[0].id : 0;
    }
    row.innerHTML =
      `<input type="checkbox" value="${cls.id}" ${checked ? "checked" : ""}>` +
      `<span class="source-class">${cls.id}: ${cls.name}</span>` +
      `<span class="map-arrow">\u2192</span>` +
      `<select class="target-id-map"></select>`;
    const sel = row.querySelector("select.target-id-map");
    mapClasses.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = String(item.id);
      opt.textContent = `${item.id}: ${item.name}`;
      sel.appendChild(opt);
    });
    sel.value = String(targetId);
    // disable dropdown when unchecked
    const cb = row.querySelector("input[type=checkbox]");
    cb.addEventListener("change", () => _syncCheckItemDisabled(row));
    _syncCheckItemDisabled(row);
    box.appendChild(row);
  });
}

async function loadClasses() {
  const modelId = $("modelSelect").value;
  if (!modelId) { toast("请先选择模型"); return; }
  const data = await api(`/api/models/${modelId}/classes`);
  const task = normalizeLabelTask(data.task || state.modelMeta[modelId]?.task);
  if ($("annotateLabelTask")) $("annotateLabelTask").value = task;
  const snap = _classUiSnapshot();
  state.sourceClasses = data.classes || [];
  await renderClassChecklist(snap);
  updateTargetClassesStatus();
  setMappingConfirmed(false, "模型类别已加载：请确认映射后再启动权重自动打标。");
}

async function applyManualTargetClasses() {
  const raw = $("targetClassesText")?.value || "";
  const names = [];
  const exists = new Set();
  raw.split(/\r?\n/).forEach((line) => {
    const name = line.trim();
    if (!name) return;
    const key = name.toLowerCase();
    if (exists.has(key)) return;
    exists.add(key);
    names.push(name);
  });
  if (!names.length) { toast("请至少输入1个输出类别（每行一个）"); return; }
  const snap = _classUiSnapshot();
  state.targetClasses = names;
  await renderClassChecklist(snap);
  updateTargetClassesStatus();
  setMappingConfirmed(false, "输出类别已变更：请重新确认映射。");
  toast(`已应用手输类别 ${state.targetClasses.length} 个`);
}

async function clearTargetClasses() {
  const snap = _classUiSnapshot();
  state.targetClasses = [];
  if ($("targetClassesText")) $("targetClassesText").value = "";
  await renderClassChecklist(snap);
  updateTargetClassesStatus();
  setMappingConfirmed(false, "输出类别已清空：请确认映射后再启动。");
  toast("已清空新场景类别，恢复使用模型原始类别");
}

async function defaultMapping() {
  if (!state.sourceClasses.length) { toast("请先加载模型类别"); return; }
  // Clear custom target classes → use model's own classes as output
  state.targetClasses = [];
  if ($("targetClassesText")) $("targetClassesText").value = "";
  // Select all, identity mapping
  await renderClassChecklist();
  // Check all — dropdown values already set correctly by renderClassChecklist (with offset)
  const box = $("classChecklist");
  if (box) {
    box.querySelectorAll(".check-item").forEach((row) => {
      const cb = row.querySelector("input[type=checkbox]");
      if (cb) cb.checked = true;
      _syncCheckItemDisabled(row);
    });
  }
  updateTargetClassesStatus();
  // Auto-confirm
  confirmClassMapping();
}

function confirmClassMapping() {
  const modelId = $("modelSelect")?.value || "";
  if (!modelId) { toast("请先选择模型"); return; }
  if (!state.sourceClasses.length) { toast("请先加载模型类别"); return; }
  const selected = getSelectedClassIds();
  if (!selected.length) { toast("请至少选择 1 个来源类别"); return; }
  state.mappingSignature = computeCurrentMappingSignature();
  setMappingConfirmed(true, `映射已确认：已选 ${selected.length} 个来源类别，可开始权重自动打标。`);
  toast("映射确认完成");
}

function buildMappingPresetPayload(name) {
  const modelId = $("modelSelect")?.value || "";
  if (!modelId) throw new Error("请先选择模型后再保存预设");
  if (!state.sourceClasses.length) throw new Error("请先加载模型类别后再保存预设");
  return {
    preset_id: `preset_${Math.random().toString(16).slice(2, 10)}`,
    name: name.trim(),
    model_id: modelId,
    model_name: $("modelSelect")?.selectedOptions?.[0]?.textContent || "",
    source_signature: sourceClassSignature(state.sourceClasses),
    source_classes: state.sourceClasses.map((x) => ({ id: x.id, name: x.name })),
    target_classes: [...state.targetClasses],
    selected_class_ids: getSelectedClassIds(),
    class_id_overrides: getClassIdOverrides(),
    updated_at: new Date().toISOString(),
  };
}

function saveClassPreset() {
  const name = $("classPresetName")?.value?.trim() || "";
  if (!name) { toast("请填写预设名称"); return; }
  try {
    const incoming = buildMappingPresetPayload(name);
    const existingIdx = state.classPresets.findIndex((x) => x.name === incoming.name);
    if (existingIdx >= 0) {
      incoming.preset_id = state.classPresets[existingIdx].preset_id;
      incoming.created_at = state.classPresets[existingIdx].created_at || incoming.updated_at;
      state.classPresets[existingIdx] = incoming;
      toast(`已更新预设: ${incoming.name}`);
    } else {
      incoming.created_at = incoming.updated_at;
      state.classPresets.unshift(incoming);
      toast(`已保存预设: ${incoming.name}`);
    }
    writeClassPresetsToStorage(state.classPresets);
    refreshClassPresetSelect();
    if ($("classPresetSelect")) $("classPresetSelect").value = incoming.preset_id;
  } catch (e) { toast(e.message); }
}

async function applyClassPreset() {
  const presetId = $("classPresetSelect")?.value || "";
  if (!presetId) { toast("请先选择预设"); return; }
  const preset = state.classPresets.find((x) => x.preset_id === presetId);
  if (!preset) { toast("预设不存在"); return; }
  const modelSel = $("modelSelect");
  if (!modelSel) return;
  if (preset.model_id && modelSel.value !== preset.model_id) {
    const exists = Array.from(modelSel.options).some((opt) => opt.value === preset.model_id);
    if (exists) { modelSel.value = preset.model_id; await loadClasses(); }
  } else if (!state.sourceClasses.length) {
    await loadClasses();
  }
  const currentSignature = sourceClassSignature(state.sourceClasses);
  if (preset.source_signature && preset.source_signature !== currentSignature) {
    toast("预设来源类别与当前模型不完全一致，已按名称/ID尽力匹配", "warning");
  }
  state.targetClasses = Array.isArray(preset.target_classes) ? [...preset.target_classes] : [];
  if ($("targetClassesText")) $("targetClassesText").value = state.targetClasses.join("\n");
  const selectedSet = new Set((preset.selected_class_ids || []).map((x) => Number(x)));
  const overrides = preset.class_id_overrides || {};
  const snap = {};
  state.sourceClasses.forEach((cls) => {
    const srcId = Number(cls.id);
    const forcedTarget = Number(overrides[String(srcId)]);
    snap[srcId] = {
      checked: selectedSet.has(srcId),
      targetId: Number.isInteger(forcedTarget) ? forcedTarget : _defaultTargetIdForSource(srcId, cls.name),
    };
  });
  await renderClassChecklist(snap);
  updateTargetClassesStatus();
  if ($("classPresetName")) $("classPresetName").value = preset.name || "";
  setMappingConfirmed(false, "映射预设已加载：请点击\"确认映射（必做）\"。");
  toast(`预设已应用: ${preset.name}`);
}

function deleteClassPreset() {
  const presetId = $("classPresetSelect")?.value || "";
  if (!presetId) { toast("请先选择预设"); return; }
  const preset = state.classPresets.find((x) => x.preset_id === presetId);
  if (!preset) { toast("预设不存在"); return; }
  if (!window.confirm(`确认删除预设「${preset.name}」吗？`)) return;
  state.classPresets = state.classPresets.filter((x) => x.preset_id !== presetId);
  writeClassPresetsToStorage(state.classPresets);
  refreshClassPresetSelect();
  toast(`已删除预设: ${preset.name}`);
}

function setClassCheckAll(checked) {
  Array.from($("classChecklist").querySelectorAll("input[type=checkbox]")).forEach((x) => { x.checked = checked; });
  Array.from($("classChecklist").querySelectorAll(".check-item")).forEach((row) => _syncCheckItemDisabled(row));
  setMappingConfirmed(false, "类别勾选已变更：请重新确认映射。");
}

function getSelectedClassIds() {
  return Array.from($("classChecklist").querySelectorAll("input[type=checkbox]:checked")).map((x) => Number(x.value));
}

function getClassIdOverrides() {
  const overrides = {};
  Array.from($("classChecklist").querySelectorAll(".check-item")).forEach((row) => {
    const cb = row.querySelector("input[type=checkbox]");
    const targetSel = row.querySelector("select.target-id-map");
    if (!cb || !targetSel || !cb.checked) return;
    const src = Number(cb.value);
    const dst = Number(targetSel.value);
    if (!Number.isInteger(src) || !Number.isInteger(dst) || dst < 0) return;
    if (src !== dst) overrides[String(src)] = dst;
  });
  return overrides;
}

/* ========== Class ID Remap ========== */
let _remapEntries = [];

async function loadClassMapping() {
  const imagesetId = $("remapImageset")?.value;
  if (!imagesetId) { toast("请先选择图片集"); return; }
  $("remapStatus").textContent = "加载中...";
  $("remapTableWrap").style.display = "none";
  $("remapResult").innerHTML = "";
  try {
    const data = await api(`/api/imagesets/${imagesetId}/class-mapping`);
    _remapEntries = data.entries || [];
    if (!_remapEntries.length) {
      $("remapStatus").textContent = "该图片集没有标注数据";
      return;
    }
    const tbody = $("remapTableBody");
    tbody.innerHTML = _remapEntries.map(e => `
      <tr>
        <td><strong>${e.class_id}</strong></td>
        <td class="remap-name">${_esc(e.name)}</td>
        <td class="remap-count">${e.count}</td>
        <td><input type="number" class="remap-new-id" data-old-id="${e.class_id}" value="${e.class_id}" min="0" step="1"></td>
      </tr>
    `).join("");
    // Highlight changed inputs
    tbody.querySelectorAll(".remap-new-id").forEach(inp => {
      inp.addEventListener("input", () => {
        inp.classList.toggle("remap-changed", Number(inp.value) !== Number(inp.dataset.oldId));
      });
    });
    $("remapTableWrap").style.display = "";
    $("remapStatus").textContent = `共 ${_remapEntries.length} 个类别，修改新ID后点击"应用ID调整"`;
  } catch (e) {
    $("remapStatus").textContent = `加载失败: ${e.message}`;
  }
}

async function applyRemap() {
  const imagesetId = $("remapImageset")?.value;
  if (!imagesetId) { toast("请先选择图片集"); return; }
  const inputs = $("remapTableBody").querySelectorAll(".remap-new-id");
  const idMapping = {};
  let hasChange = false;
  for (const inp of inputs) {
    const oldId = Number(inp.dataset.oldId);
    const newId = Number(inp.value);
    if (!Number.isInteger(newId) || newId < 0) {
      toast(`ID ${oldId} 的新值无效`, "error");
      inp.focus();
      return;
    }
    idMapping[String(oldId)] = newId;
    if (oldId !== newId) hasChange = true;
  }
  if (!hasChange) { toast("没有任何 ID 变更"); return; }
  // Check duplicates: allow same-name classes to merge, reject different-name conflicts
  const targetToNames = {};
  for (const inp of inputs) {
    const oldId = inp.dataset.oldId;
    const newId = Number(inp.value);
    const name = inp.closest("tr")?.querySelectorAll("td")[1]?.textContent?.trim() || "";
    if (!targetToNames[newId]) targetToNames[newId] = new Set();
    targetToNames[newId].add(name);
  }
  for (const [nid, names] of Object.entries(targetToNames)) {
    if (names.size > 1) {
      toast(`目标 ID ${nid} 对应多个不同类名: ${[...names].join(", ")}，只有同名类别才能合并`, "error");
      return;
    }
  }
  if (!window.confirm("确认应用 ID 调整？此操作会直接修改标签文件，不可撤销。")) return;
  $("remapStatus").textContent = "应用中...";
  try {
    const result = await api(`/api/imagesets/${imagesetId}/remap-classes`, {
      method: "POST",
      body: { id_mapping: idMapping },
    });
    $("remapResult").innerHTML =
      `<div class="status-box">完成：修改 ${result.files_modified} 个文件，${result.lines_modified} 行标签，${result.new_class_count} 个类别</div>`;
    toast("ID 调整完成", "success");
    await loadClassMapping();
    await refreshImagesetSelects();
  } catch (e) {
    $("remapStatus").textContent = `应用失败: ${e.message}`;
    toast(`应用失败: ${e.message}`, "error");
  }
}
