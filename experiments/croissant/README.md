# Croissant Metadata and HuggingFace Upload

This directory contains the [ML Commons Croissant v1.0](http://mlcommons.org/croissant/)
metadata descriptors for the two datasets released alongside the NeurIPS 2026
paper. Both files have been validated against the `mlcroissant` reference
implementation and conform to the Croissant + RAI (Responsible AI) extension.

## Files

| File | Dataset | HuggingFace repo | Size |
|---|---|---|---|
| [training.json](training.json) | 25,708 training result JSON files | `vikpant/coopetition-gym-v1` | ~900 MB tarball |
| [audit.json](audit.json) | 1,116 behavioral audit JSON files | `vikpant/coopetition-gym-audit` | 605 KB tarball |

## What's inside each Croissant file

Both files declare:

- **Core metadata** — `name`, `description`, `citation`, `license` (CC-BY-4.0),
  `url`, `version`, `datePublished`, creators, publisher, keywords.
- **RAI (Responsible AI) extension** — `dataCollection`, `dataPreprocessingProtocol`,
  `dataBiases`, `personalSensitiveInformation`, `useCases`, `dataLimitation`,
  `dataSocialImpact`, `dataReleaseMaintenancePlan`.
- **Distribution** — one `FileObject` per tarball plus `FileSet` declarations
  for the JSON files within.
- **Record sets** — field-level schema for each JSON file, with JSONPath
  extractions.

The `sha256` fields are placeholders (`TBD-COMPUTED-AT-UPLOAD-TIME`) because
the final tarball checksums must be recomputed after any last-minute content
changes. These values must be updated before upload (see step 2 below).

## Upload Workflow

### 1. HuggingFace authentication (one time)

```bash
pip install huggingface_hub  # already in the dev venv
huggingface-cli login
```

Enter your HuggingFace access token when prompted. Generate one at
`https://huggingface.co/settings/tokens` with **write** scope.

### 2. Prepare the tarballs and compute checksums

```bash
# Audit tarball already exists
AUDIT_TARBALL="/path/to/behavioral_audits.tar.gz"
sha256sum "$AUDIT_TARBALL"

# Training tarball: create from the unified dataset
cd /path/to/unified_dataset
tar -czf /tmp/unified_dataset.tar.gz \
    baseline_integrated/ \
    ablation_private/ \
    ablation_cooperative/ \
    case_study/ \
    france_bonus_isac_integrated/ \
    local_bonus/ \
    network_sensitivity/
sha256sum /tmp/unified_dataset.tar.gz
```

Update the `sha256` fields in `training.json` and `audit.json` with the
computed hashes.

### 3. Create the HuggingFace repositories

```bash
huggingface-cli repo create coopetition-gym-v1 --type dataset
huggingface-cli repo create coopetition-gym-audit --type dataset
```

### 4. Upload tarballs and Croissant metadata

```bash
# Training dataset
huggingface-cli upload vikpant/coopetition-gym-v1 \
    /tmp/unified_dataset.tar.gz \
    unified_dataset.tar.gz \
    --repo-type dataset

huggingface-cli upload vikpant/coopetition-gym-v1 \
    experiments/croissant/training.json \
    croissant.json \
    --repo-type dataset

# Audit dataset
huggingface-cli upload vikpant/coopetition-gym-audit \
    /path/to/behavioral_audits.tar.gz \
    behavioral_audits.tar.gz \
    --repo-type dataset

huggingface-cli upload vikpant/coopetition-gym-audit \
    experiments/croissant/audit.json \
    croissant.json \
    --repo-type dataset
```

HuggingFace Hub auto-detects `croissant.json` at the repo root and exposes it
on the dataset page under a "Croissant metadata" section.

### 5. Write a HuggingFace README for each dataset

Each dataset repo needs a `README.md` at its root. The README appears as the
dataset landing page. Use the `DATASHEET.md` from the repository root as a
starting point and trim to the single dataset.

```bash
huggingface-cli upload vikpant/coopetition-gym-v1 \
    /path/to/training_dataset_README.md README.md \
    --repo-type dataset

huggingface-cli upload vikpant/coopetition-gym-audit \
    /path/to/audit_dataset_README.md README.md \
    --repo-type dataset
```

### 6. Verify the upload

```bash
# Check that the datasets resolve with Croissant metadata attached
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for repo in ["vikpant/coopetition-gym-v1", "vikpant/coopetition-gym-audit"]:
    info = api.dataset_info(repo)
    print(f"{repo}: {info.id}, private={info.private}, downloads={info.downloads}")
PY
```

Then visit the dataset page in a browser and confirm the Croissant button
appears in the sidebar.

### 7. Cross-link in the paper and repository

After upload:

1. **OpenReview supplementary**: cite the two HuggingFace URLs in the
   submission's data availability statement.
2. **arXiv**: include the two URLs in the arXiv abstract submission.
3. **Repository README**: update [../../REPRODUCE.md](../../REPRODUCE.md) to
   confirm the HuggingFace URLs (they are already referenced but check the
   download instructions resolve).

## Validation

To re-validate the Croissant files locally:

```bash
python - <<'PY'
import mlcroissant
for path in ["experiments/croissant/training.json",
             "experiments/croissant/audit.json"]:
    ds = mlcroissant.Dataset(path)
    print(f"{path}: {ds.metadata.name} v{ds.metadata.version}")
    print(f"  distribution: {len(ds.metadata.distribution)} items")
    print(f"  recordSet: {len(ds.metadata.record_sets)} sets")
PY
```

Both files currently produce one harmless warning about `samplingRate` and
`equivalentProperty` context keys; these are optional Croissant context
declarations that neither dataset uses.
