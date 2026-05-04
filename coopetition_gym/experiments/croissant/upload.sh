#!/usr/bin/env bash
# Upload Coopetition-Gym v1 logs to the unified HuggingFace dataset
# vikpant/coopetition-gym-logs.
#
# The dataset is organized as four top-level subdirectories of JSONL shards:
#   training_runs/         training corpus (~17,930 records across 949 JSONL shards)
#   behavioral_audit/      behavioral audit corpus (1,116 records across 2 shards
#                          plus a manifest CSV)
#   case_study_calibration/ four case study calibration JSONL files
#   lr_ablation/           controlled critic-learning-rate ablation runs
#
# This script assumes:
#   * ``hf auth login`` has already been run with a write-scope token.
#   * Local source directories for the four subdirectory contents are pointed at
#     by environment variables (or default to layouts under ${SOURCE_ROOT}).
#
# Environment variables:
#   HF_USER                  HuggingFace username (default "vikpant")
#   HF_REPO                  Repo name (default "coopetition-gym-logs")
#   HF_PRIVATE               "1" to create the repo private (default "0", public)
#   SOURCE_ROOT              Local root for the four subdirectories
#                            (default ".claude/release_payload/")
#
# Usage:
#   export SOURCE_ROOT=/path/to/release_payload
#   bash experiments/croissant/upload.sh
#
# The script:
#   1. Verifies login state.
#   2. Creates the dataset repo if absent (idempotent).
#   3. Uploads README.md and croissant.json from this directory's authoritative
#      copies (papers/neurips_ed_2026/croissant.json and
#      coopetition_gym/experiments/croissant/hf_readme_*.md).
#   4. Uploads each of the four subdirectory contents.
#   5. Prints a verification summary.

set -euo pipefail

HF_USER="${HF_USER:-vikpant}"
HF_REPO="${HF_REPO:-coopetition-gym-logs}"
HF_PRIVATE="${HF_PRIVATE:-0}"
SOURCE_ROOT="${SOURCE_ROOT:-.claude/release_payload}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CROISSANT_FILE="$REPO_ROOT/papers/neurips_ed_2026/croissant.json"
README_FILE="$REPO_ROOT/coopetition_gym/experiments/croissant/hf_readme_training.md"

require() {
    command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }
}

require hf
require python3

# Check login state via whoami; exit if not logged in.
hf auth whoami >/dev/null 2>&1 || {
    echo "Not logged in. Run: hf auth login"
    exit 1
}

[ -f "$CROISSANT_FILE" ] || { echo "missing canonical Croissant: $CROISSANT_FILE"; exit 1; }
[ -f "$README_FILE" ]    || { echo "missing README: $README_FILE"; exit 1; }
[ -d "$SOURCE_ROOT" ]    || { echo "missing SOURCE_ROOT: $SOURCE_ROOT"; exit 1; }

echo "=== Step 1: ensure dataset repo exists ==="
PRIVATE_FLAG=""
if [ "$HF_PRIVATE" = "1" ]; then
    PRIVATE_FLAG="--private"
    echo "  creating PRIVATE repo (HF_PRIVATE=1)"
else
    echo "  creating PUBLIC repo (HF_PRIVATE=0)"
fi
hf repo create "$HF_USER/$HF_REPO" --type dataset $PRIVATE_FLAG --yes 2>&1 | tail -3 \
    || echo "  (repo may already exist, continuing)"

echo "=== Step 2: upload README.md and croissant.json ==="
hf upload "$HF_USER/$HF_REPO" "$README_FILE"    README.md       --repo-type dataset
hf upload "$HF_USER/$HF_REPO" "$CROISSANT_FILE" croissant.json  --repo-type dataset

echo "=== Step 3: upload subdirectory payloads ==="
for sub in training_runs behavioral_audit case_study_calibration lr_ablation; do
    if [ -d "$SOURCE_ROOT/$sub" ]; then
        echo "  uploading $sub/ ..."
        hf upload "$HF_USER/$HF_REPO" "$SOURCE_ROOT/$sub" "$sub" --repo-type dataset
    else
        echo "  skipping $sub/ (not present at $SOURCE_ROOT/$sub)"
    fi
done

echo "=== Step 4: verify ==="
python3 - "$HF_USER" "$HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
user, repo = sys.argv[1], sys.argv[2]
api = HfApi()
info = api.dataset_info(f"{user}/{repo}")
print(f"  {info.id}: private={info.private}, sha={info.sha[:12] if info.sha else 'None'}, files={len(info.siblings)}")
PY

echo "=== Complete ==="
echo "Dataset URL: https://huggingface.co/datasets/$HF_USER/$HF_REPO"
if [ "$HF_PRIVATE" = "1" ]; then
    cat <<'MSG'

Dataset uploaded as PRIVATE. To make it public when ready:

  curl -X PUT \
    -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    -H "Content-Type: application/json" \
    -d '{"private": false}' \
    "https://huggingface.co/api/datasets/$HF_USER/$HF_REPO/settings"
MSG
fi
