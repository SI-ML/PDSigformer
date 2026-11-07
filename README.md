# PDSigformer

Official implementation of **PDSigformer**, accepted at ICIMH 2026, for subject-independent Parkinson's disease classification from wearable inertial signals.

Built on [Medformer](https://github.com/DL4mHealth/Medformer), PDSigformer combines a TimesNet frontend, residual fusion with the input signal, and multi-granularity attention. This repository includes 13 baselines and four ablation configurations (R1–R4).

## Release Status

**Checkpoints and publicly releasable processed data will be uploaded in a future update. Download links will be added here.** Training and evaluation require the corresponding data artifacts.

## Requirements

Validated with Python 3.10, PyTorch 2.8.0, CUDA 12.8, and NVIDIA RTX 4090 GPUs.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-pdsigformer.txt
```

To run the TeCh baseline, also run `python scripts/fetch_tech.py`. Its implementation is fetched separately from a pinned revision; see [third-party sources](third_party/README.md). Exact dependency versions are recorded in [requirements-pdsigformer-lock.txt](requirements-pdsigformer-lock.txt).

## Datasets

Experiments use a fixed subject-independent split (`split42`) and training seeds **42, 43, and 44**. Changing the training seed does not change the data split.

| Dataset | Activity | Input shape | Train / Valid / Test subjects |
|---|---|---|---|
| PADS | CrossArms | 976 × 6 | 215 / 70 / 70 |
| Shimmer | DRINK | 4096 × 6 | 80 / 26 / 24 |

Place the released artifact bundle in `artifacts/`. The expected layout and import instructions are documented in [ARTIFACTS.md](docs/ARTIFACTS.md); preprocessing and evaluation details are in [PROTOCOL.md](docs/PROTOCOL.md).

## Run Experiments

List configurations:

```bash
python run_wearable.py list
```

Train PDSigformer from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python run_wearable.py train \
  --profile shimmer_r4 --seed 42 --output outputs/train_shimmer_r4_s42
```

Evaluate the resulting checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python run_wearable.py eval \
  --profile shimmer_r4 --seed 42 \
  --checkpoint outputs/train_shimmer_r4_s42/best.pt \
  --output outputs/test_shimmer_r4_s42
```

After the historical checkpoint bundle is released, omit `--checkpoint` to evaluate the checkpoint recorded in the experiment manifest. Use `pads_r4` for PADS or another profile from `list` for a baseline or ablation.

All hyperparameters are specified in [configs/paper.json](configs/paper.json). Training uses up to 100 epochs with patience 100 and selects `best.pt` by validation Macro-F1. Outputs include the effective configuration, per-epoch metrics, predictions, and checkpoint. Use a new output directory for each run.

Run all 34 configurations with three training seeds, or evaluate all historical checkpoints:

```bash
bash scripts/pdsigformer/train_all.sh 0,1 outputs/training_all
bash scripts/pdsigformer/evaluate_all.sh 0,1 outputs/evaluation_all
```

The experiments reference 102 unique checkpoints. R4 is shared by the main comparison and the ablation table. Keep the configured evaluation batch size and sample order, since TimesNet's period selection depends on the batch.

## Validation

All 34 configurations passed two independent five-epoch training and checkpoint-reload tests with seed 42. Under the default settings, 33 pairs matched exactly; PADS Reformer showed numerical differences. An additional paired Reformer test matched exactly with deterministic algorithms enabled. Full 100-epoch retraining to reproduce all reported paper scores has not yet been verified.

## Acknowledgements

We thank [Medformer](https://github.com/DL4mHealth/Medformer), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), and the TeCh authors. The upstream Medformer Python files are preserved. The original README is retained as [README_MEDFORMER.md](README_MEDFORMER.md); source versions and changes are documented in [CHANGES.md](docs/CHANGES.md).
