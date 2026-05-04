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

async function uploadFolderAsImageset() {
  const files = Array.from($("folderInput").files || []);
  if (!files.length) { toast("请先选择目录"); return; }
  const form = new FormData();
  appendFolderFiles(form, files);
  form.append("imageset_name", `folder_${Date.now()}`);
  $("annotateStatus").textContent = "目录上传中...";
  try {
    const data = await api("/api/imagesets/upload-folder", { method: "POST", body: form });
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
  const form = new FormData();
  appendFolderFiles(form, files);
  form.append("imageset_name", `folder_${Date.now()}`);
  $("qwenStatus").textContent = "目录上传中...";
  try {
    const data = await api("/api/imagesets/upload-folder", { method: "POST", body: form });
    $("qwenStatus").textContent = datasetUploadSummary(data);
    await refreshImagesetSelects();
    if ($("aiImagesetSelect")) $("aiImagesetSelect").value = data.imageset_id;
    toast("目录图片集已就绪", "success");
  } catch (e) {
    $("qwenStatus").textContent = e.message;
    toast(`目录上传失败: ${e.message}`, "error");
  }
}
