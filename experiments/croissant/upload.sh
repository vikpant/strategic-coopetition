#!/usr/bin/env bash
# Upload Coopetition-Gym v1 training and audit datasets to HuggingFace Hub.
#
# This script assumes:
#   * ``huggingface-cli login`` has already been run with a write-scope token.
#   * ``TRAINING_TARBALL`` environment variable (or the first positional
#     argument) points at the assembled training dataset tarball.
#   * ``AUDIT_TARBALL`` environment variable (or the second positional
#     argument) points at the behavioral audit tarball.
#
# Environment variables:
#   TRAINING_TARBALL       Path to unified training dataset tarball
#   AUDIT_TARBALL          Path to behavioral audit tarball
#   HF_PRIVATE             "1" to create private repos (default "1" — flip to
#                          public only after arXiv posting establishes priority)
#   HF_USER                HuggingFace username (default "vikpant")
#
# Usage:
#   export TRAINING_TARBALL=/tmp/unified_dataset.tar.gz
#   export AUDIT_TARBALL=.claude/experiments/neurips_ablation/behavioral_audits.tar.gz
#   bash experiments/croissant/upload.sh
#
# The script performs these steps idempotently:
#   1. Computes SHA-256 for both tarballs.
#   2. Patches the Croissant JSON files with real hashes.
#   3. Creates the two HuggingFace dataset repos (private by default).
#   4. Uploads tarballs, Croissant metadata, and READMEs.
#   5. Prints a verification summary.

set -euo pipefail

HF_USER="${HF_USER:-vikpant}"
HF_PRIVATE="${HF_PRIVATE:-1}"
TRAINING_TARBALL="${TRAINING_TARBALL:-${1:-}}"
AUDIT_TARBALL="${AUDIT_TARBALL:-${2:-}}"

CROISSANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CROISSANT="$CROISSANT_DIR/training.json"
AUDIT_CROISSANT="$CROISSANT_DIR/audit.json"
TRAIN_README="$CROISSANT_DIR/hf_readme_training.md"
AUDIT_README="$CROISSANT_DIR/hf_readme_audit.md"

require() {
    command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }
}

require huggingface-cli
require sha256sum
require python3

if [ -z "$TRAINING_TARBALL" ] || [ ! -f "$TRAINING_TARBALL" ]; then
    echo "TRAINING_TARBALL not set or file missing: '$TRAINING_TARBALL'"
    exit 1
fi
if [ -z "$AUDIT_TARBALL" ] || [ ! -f "$AUDIT_TARBALL" ]; then
    echo "AUDIT_TARBALL not set or file missing: '$AUDIT_TARBALL'"
    exit 1
fi

# Check login state via whoami; exit if not logged in.
huggingface-cli whoami >/dev/null 2>&1 || {
    echo "Not logged in. Run: huggingface-cli login"
    exit 1
}

echo "=== Step 1: compute SHA-256 ==="
TRAINING_SHA="$(sha256sum "$TRAINING_TARBALL" | awk '{print $1}')"
AUDIT_SHA="$(sha256sum "$AUDIT_TARBALL" | awk '{print $1}')"
echo "  training: $TRAINING_SHA"
echo "  audit:    $AUDIT_SHA"

echo "=== Step 2: patch Croissant files with actual hashes ==="
python3 - "$TRAIN_CROISSANT" "$TRAINING_SHA" <<'PY'
import json, sys
path, new_sha = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
for dist in data.get("distribution", []):
    if dist.get("@type") == "cr:FileObject" and dist.get("sha256") == "TBD-COMPUTED-AT-UPLOAD-TIME":
        dist["sha256"] = new_sha
with open(path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  patched {path}")
PY
python3 - "$AUDIT_CROISSANT" "$AUDIT_SHA" <<'PY'
import json, sys
path, new_sha = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
for dist in data.get("distribution", []):
    if dist.get("@type") == "cr:FileObject" and dist.get("sha256") == "TBD-COMPUTED-AT-UPLOAD-TIME":
        dist["sha256"] = new_sha
with open(path, "w") as f:
    json.dump(data, f, indent=2)
print(f"  patched {path}")
PY

PRIVATE_FLAG=""
if [ "$HF_PRIVATE" = "1" ]; then
    PRIVATE_FLAG="--private"
    echo "=== Step 3: create PRIVATE HuggingFace repos ==="
    echo "  (flip to public later via: huggingface-cli repo visibility REPO --type dataset)"
else
    echo "=== Step 3: create PUBLIC HuggingFace repos ==="
fi

huggingface-cli repo create coopetition-gym-v1 --type dataset $PRIVATE_FLAG --yes 2>&1 | tail -3 || echo "  (repo may already exist, continuing)"
huggingface-cli repo create coopetition-gym-audit --type dataset $PRIVATE_FLAG --yes 2>&1 | tail -3 || echo "  (repo may already exist, continuing)"

echo "=== Step 4: upload training dataset ==="
huggingface-cli upload "$HF_USER/coopetition-gym-v1" \
    "$TRAINING_TARBALL" unified_dataset.tar.gz --repo-type dataset
huggingface-cli upload "$HF_USER/coopetition-gym-v1" \
    "$TRAIN_CROISSANT" croissant.json --repo-type dataset
huggingface-cli upload "$HF_USER/coopetition-gym-v1" \
    "$TRAIN_README" README.md --repo-type dataset

echo "=== Step 5: upload audit dataset ==="
huggingface-cli upload "$HF_USER/coopetition-gym-audit" \
    "$AUDIT_TARBALL" behavioral_audits.tar.gz --repo-type dataset
huggingface-cli upload "$HF_USER/coopetition-gym-audit" \
    "$AUDIT_CROISSANT" croissant.json --repo-type dataset
huggingface-cli upload "$HF_USER/coopetition-gym-audit" \
    "$AUDIT_README" README.md --repo-type dataset

echo "=== Step 6: verify ==="
python3 - "$HF_USER" <<'PY'
import sys
from huggingface_hub import HfApi
user = sys.argv[1]
api = HfApi()
for name in ("coopetition-gym-v1", "coopetition-gym-audit"):
    info = api.dataset_info(f"{user}/{name}")
    print(f"  {info.id}: private={info.private}, sha={info.sha[:12] if info.sha else 'None'}")
PY

echo "=== Complete ==="
if [ "$HF_PRIVATE" = "1" ]; then
    cat <<'MSG'

Datasets uploaded as PRIVATE.

Before making them public (recommended: same day as arXiv posting):

  huggingface-cli repo visibility vikpant/coopetition-gym-v1 --type dataset
  huggingface-cli repo visibility vikpant/coopetition-gym-audit --type dataset

For reviewer access during NeurIPS review, share the repo URL and grant
read access via the HuggingFace web UI (Settings → Access requests).
MSG
fi
