function appendFolderFiles(form, files) {
  files.forEach((file) => form.append("files", file, file.webkitRelativePath || file.name));
}

function datasetUploadSummary(data) {
  const warnParts = [];
  if ((data.labels_unmatched || 0) > 0) warnParts.push(`未匹配txt ${data.labels_unmatched}`);
  if ((data.duplicate_label_name_count || 0) > 0) warnParts.push(`重复txt ${data.duplicate_label_name_count}`);
  const warnText = warnParts.length ? `; ${warnParts.join(", ")}` : "";
  return `上传成功：${data.name}，图 ${data.image_count}，txt ${data.labels_imported || 0}，类别 ${data.class_names_imported || 0}${warnText}`;
}

function _folderUploadTotalBytes(files) {
  return files.reduce((sum, file) => sum + Number(file.size || 0), 0);
}

function _folderUploadProgress(statusEl, label, state) {
  const totalBytes = Math.max(1, Number(state.totalBytes || 0));
  const loaded = Number(state.doneBytes || 0) + Number(state.fileLoaded || 0);
  const current = state.currentFile || "";
  const phase = `文件 ${state.fileIndex}/${state.fileCount} · ${current} · ${formatBytes(Math.min(loaded, totalBytes))} / ${formatBytes(totalBytes)}`;
  renderUploadProgress(statusEl, {
    label,
    loaded: Math.min(loaded, totalBytes),
    total: totalBytes,
    phase,
  });
}

async function uploadFolderFilesWithSession(files, imagesetName, statusEl) {
  const totalBytes = _folderUploadTotalBytes(files);
  const label = `上传目录 ${files.length} 个文件`;
  renderUploadProgress(statusEl, { label, loaded: 0, total: Math.max(1, totalBytes), phase: "创建上传会话..." });
  const session = await api("/api/imagesets/upload-folder/session", {
    method: "POST",
    body: { imageset_name: imagesetName, total_files: files.length, total_bytes: totalBytes },
  });
  let doneBytes = 0;
  for (let idx = 0; idx < files.length; idx += 1) {
    const file = files[idx];
    const relPath = file.webkitRelativePath || file.name;
    const form = new FormData();
    form.append("relative_path", relPath);
    form.append("file", file, file.name);
    await uploadFormWithProgress(`/api/imagesets/upload-folder/session/${session.session_id}/file`, {
      method: "POST",
      body: form,
      statusEl,
      label,
      onProgress: (progress) => _folderUploadProgress(statusEl, label, {
        totalBytes,
        doneBytes,
        fileLoaded: progress.loaded,
        fileIndex: idx + 1,
        fileCount: files.length,
        currentFile: relPath,
      }),
      onUploadComplete: () => _folderUploadProgress(statusEl, label, {
        totalBytes,
        doneBytes: doneBytes + Number(file.size || 0),
        fileLoaded: 0,
        fileIndex: idx + 1,
        fileCount: files.length,
        currentFile: relPath,
      }),
    });
    doneBytes += Number(file.size || 0);
  }
  renderUploadProgress(statusEl, {
    label,
    loaded: Math.max(1, totalBytes),
    total: Math.max(1, totalBytes),
    phase: "上传完成，正在整理图片和标签...",
  });
  return api(`/api/imagesets/upload-folder/session/${session.session_id}/finish`, { method: "POST" });
}

async function uploadFolderAsImageset() {
  const files = Array.from($("folderInput").files || []);
  if (!files.length) { toast("请先选择目录"); return; }
  const imagesetName = `folder_${Date.now()}`;
  $("annotateStatus").textContent = "目录上传中...";
  try {
    const data = await uploadFolderFilesWithSession(files, imagesetName, "annotateStatus");
    $("annotateStatus").textContent = datasetUploadSummary(data);
    await refreshImagesetSelects();
    $("annotateImageset").value = data.imageset_id;
    if ($("aiImagesetSelect")) $("aiImagesetSelect").value = data.imageset_id;
    toast("目录图片集已就绪", "success");
  } catch (e) {
    $("annotateStatus").textContent = e.message;
    toast(`目录上传失败: ${e.message}`, "error");
  }
}

async function uploadAiFolderAsImageset() {
  const files = Array.from($("aiFolderInput")?.files || []);
  if (!files.length) { toast("请先选择目录"); return; }
  const imagesetName = `folder_${Date.now()}`;
  $("qwenStatus").textContent = "目录上传中...";
  try {
    const data = await uploadFolderFilesWithSession(files, imagesetName, "qwenStatus");
    $("qwenStatus").textContent = datasetUploadSummary(data);
    await refreshImagesetSelects();
    if ($("aiImagesetSelect")) $("aiImagesetSelect").value = data.imageset_id;
    toast("目录图片集已就绪", "success");
  } catch (e) {
    $("qwenStatus").textContent = e.message;
    toast(`目录上传失败: ${e.message}`, "error");
  }
}
