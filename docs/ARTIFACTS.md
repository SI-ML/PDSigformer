# Data and artifacts

```text
artifacts/
  objects/sha256/<prefix>/<SHA>.pt or .pth
  manifests/baseline78_unified_v4.json
  manifests/pds24.json
  data_snapshots/pds_shimmer/
  data_snapshots/pds_pads/
  data_snapshots/baseline_common/prepared/<task>/
  original_evidence/...
  commands/                 # Archived numerical compatibility modules
  native_compat/            # Plus runtime directories listed in the manifest
  IMPORT_RECEIPT.json
```

The import script copies required files and verifies their hashes without modifying the source archive. Missing PDS Train/Valid arrays are recovered from baseline archives only when file hashes, sample identities, and ordering match exactly. `IMPORT_RECEIPT.json` records their provenance.

PDS caches contain `manifest.json`, `train_X.npy`, `train_y.npy`, `train_windows.csv`, and corresponding Valid/Test files. Inputs are float32 `[N,T,C]`; labels are int64 `[N]`. CSV mappings preserve fields such as `loader_index`, `label`, `subject_id`, and `window_id`. Baseline caches use `<split>/X.npy`, `y.npy`, and `identities.csv`, with metadata in `task.json`.

Data are not bundled with the source. Raw-data windowing, class filtering, and subject assignments must follow the archived preprocessing protocol. Matching array shapes alone do not establish equivalence: file hashes and identity mappings must also match. Training requires Train/Valid caches, but no historical checkpoints.

## Historical inference runtimes

Historical TimesNet/Autoformer Test runs use the cuFFT 10.9 numerical path; Reformer uses frozen CUDA 11.8 GELU PTX. All run under PyTorch 2.8.0+cu128. Runtime files are listed in the baseline manifest and verified before loading. Runtime binaries are imported from the artifact bundle rather than distributed with source. The validated Reformer GPU architecture is SM89.

## Records

Public source includes configurations, hashes, table mappings, aggregate metrics, and validation status. Weights, data, sample identities and predictions, server paths, and complete original evidence remain in the artifact store. `checkpoints/` provides 108 readable references to 102 unique objects. Each run uses a new `outputs/` directory so failed attempts are not overwritten.
