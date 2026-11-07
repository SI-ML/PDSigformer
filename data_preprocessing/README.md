# Data preprocessing

`PADS_preprocessing.py` and `Shimmer_preprocessing.py` are empty placeholders. Implementations will be released later.

The planned workflow follows Medformer's offline preprocessing and dataset-loader structure:

1. Select the target activity and HC/PD labels.
2. Preserve the fixed `split42` Train/Valid/Test subject assignments.
3. Apply the archived windowing/padding rules and per-window, per-channel `StandardScaler`.
4. Export arrays, labels, subject/window mappings, and file hashes for `data_provider/wearable_loader.py`.

Current experiments load frozen prepared arrays and do not call these placeholders. See the [protocol](../docs/PROTOCOL.md) and [data layout](../docs/ARTIFACTS.md).
