# Croissant Metadata and HuggingFace Upload

This directory contains [ML Commons Croissant v1.0](http://mlcommons.org/croissant/) metadata descriptors and HuggingFace dataset README sources for the unified `vikpant/coopetition-gym-logs` dataset that accompanies the Coopetition-Gym v1 release.

## Files

| File | Role |
|---|---|
| [hf_readme_training.md](hf_readme_training.md) | Training-corpus-focused README (used as the canonical dataset README on HuggingFace) |
| [hf_readme_audit.md](hf_readme_audit.md) | Behavioral-audit-corpus-focused README (alternative single-corpus framing) |
| [training.json](training.json) | Legacy split-manifest authored when v1 was planned to ship as two separate datasets. Retained for historical reference only. |
| [audit.json](audit.json) | Legacy split-manifest, same caveat. |
| [upload.sh](upload.sh) | Uploader script for the unified `coopetition-gym-logs` dataset |

The **canonical authoritative Croissant manifest** for the unified dataset lives at `papers/neurips_ed_2026/croissant.json` (5 filesets: `training_runs`, `behavioral_audit`, `case_study_calibration`, `tier_1_5_2d_slcd`, `lr_ablation`). The local files `training.json` and `audit.json` in this directory describe an earlier two-dataset layout that was superseded; they are left in place for provenance but are not the source of truth for the deployed dataset.

## Deployed dataset structure

The unified `vikpant/coopetition-gym-logs` dataset on HuggingFace is organized as four (current) top-level subdirectories of JSONL shards:

```
vikpant/coopetition-gym-logs/
├── README.md                       # rendered as the dataset landing page
├── croissant.json                  # canonical Croissant 1.0 manifest
├── LICENSE.md                      # CC-BY-4.0
├── training_runs/                  # 949 JSONL shards (~4.7 GB total)
│   └── training_runs_NNNN.jsonl    # one record per training run
├── behavioral_audit/               # 2 JSONL shards plus a manifest CSV
│   ├── behavioral_audit_NNNN.jsonl
│   └── behavioral_audit_manifest.csv
├── case_study_calibration/         # 4 case study JSONL files
│   ├── apache_http_server.jsonl
│   ├── apple_ios_app_store.jsonl
│   ├── renault_nissan_alliance.jsonl
│   └── samsung_sony_slcd.jsonl
└── lr_ablation/                    # 1 JSONL shard plus a manifest CSV
    ├── lr_ablation_NNNN.jsonl
    └── lr_ablation_manifest.csv
```

A planned future subdirectory `tier_1_5_2d_slcd/` is described in the Croissant manifest but not yet uploaded.

## What's inside the canonical Croissant manifest

The manifest at `papers/neurips_ed_2026/croissant.json` declares:

- **Core metadata**: `name`, `description`, `citation`, `license` (CC-BY-4.0), `url`, `version`, `datePublished`, creators, publisher, keywords.
- **RAI (Responsible AI) extension**: `dataCollection`, `dataPreprocessingProtocol`, `dataBiases`, `personalSensitiveInformation`, `useCases`, `dataLimitation`, `dataSocialImpact`, `dataReleaseMaintenancePlan`.
- **Distribution**: one `cr:FileSet` per subdirectory, each with `includes: <subdir>/*.jsonl` and `encodingFormat: application/jsonlines`.
- **Record sets**: per-corpus field-level schema with JSONPath extractions for the JSONL records.

JSONL shards are line-delimited JSON; one record per line. No tarball extraction step is needed.

## Upload Workflow

### 1. HuggingFace authentication (one time)

```bash
pip install huggingface_hub  # already in the dev venv
hf auth login
```

Enter your HuggingFace access token when prompted. Generate one at <https://huggingface.co/settings/tokens> with **write** scope.

### 2. Run the uploader

```bash
export SOURCE_ROOT=/path/to/release_payload
bash experiments/croissant/upload.sh
```

The script idempotently:
1. Verifies HuggingFace login state.
2. Creates the `vikpant/coopetition-gym-logs` dataset repo if absent (public by default; set `HF_PRIVATE=1` for private).
3. Uploads the canonical README.md and croissant.json.
4. Uploads each of the four subdirectory contents from `$SOURCE_ROOT`.
5. Prints a verification summary via `huggingface_hub.HfApi.dataset_info`.

### 3. Toggle visibility (if uploaded private)

```bash
curl -X PUT \
    -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    -H "Content-Type: application/json" \
    -d '{"private": false}' \
    "https://huggingface.co/api/datasets/vikpant/coopetition-gym-logs/settings"
```

### 4. Verify reviewer-fetch readiness

```bash
# Anonymous (no Authorization header) sanity checks
curl -o /dev/null -w "%{http_code}\n" "https://huggingface.co/api/datasets/vikpant/coopetition-gym-logs"
curl -o /dev/null -w "%{http_code}\n" "https://huggingface.co/datasets/vikpant/coopetition-gym-logs/raw/main/README.md"
curl -o /dev/null -w "%{http_code}\n" "https://huggingface.co/datasets/vikpant/coopetition-gym-logs/raw/main/croissant.json"
curl -o /dev/null -w "%{http_code}\n" "https://huggingface.co/api/datasets/vikpant/coopetition-gym-logs/croissant"
```

All four should return `200`. The fourth (`/api/.../croissant`) is the HuggingFace auto-Croissant endpoint that the NeurIPS reviewer portal uses for automated metadata fetch.

## Local Croissant validation

```bash
python - <<'PY'
import mlcroissant
ds = mlcroissant.Dataset("papers/neurips_ed_2026/croissant.json")
print(f"name: {ds.metadata.name} v{ds.metadata.version}")
print(f"distribution: {len(ds.metadata.distribution)} filesets")
print(f"recordSet: {len(ds.metadata.record_sets)} sets")
PY
```

Or with `cffconvert`-style direct validation:

```bash
mlcroissant validate --jsonld papers/neurips_ed_2026/croissant.json
```
