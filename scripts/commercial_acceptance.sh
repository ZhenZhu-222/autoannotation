#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Python syntax =="
python -m compileall -q app

echo "== Frontend syntax =="
if command -v node >/dev/null 2>&1; then
  node --check front/ui_utils.js
  node --check front/api.js
  node --check front/dataset_upload.js
  node --check front/imageset_preview.js
  node --check front/gallery.js
  node --check front/class_mapping.js
  node --check front/annotate.js
  node --check front/qwen_annotate.js
  node --check front/jobs_history.js
  node --check front/refine.js
  node --check front/train.js
  node --check front/preview.js
  node --check front/app.js
else
  echo "node not found, skip frontend syntax check"
fi

echo "== Diff whitespace =="
git diff --check

echo "== Tests =="
PYTHONPATH=. pytest -q

echo "commercial acceptance passed"
