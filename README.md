# PDSigformer

Official implementation of **PDSigformer**, accepted at ICIMH 2026, for subject-independent Parkinson's disease classification from wearable inertial signals.

**Checkpoints and publicly releasable data will be uploaded later.**

## Installation

Python 3.10 and PyTorch 2.8.0 (CUDA 12.8).

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-pdsigformer.txt
```

## Data

We use PADS (CrossArms) and Shimmer (DRINK) with the fixed subject-independent split `split42`. Place the released data in `artifacts/`; see the [data layout](docs/ARTIFACTS.md). [Preprocessing scripts](data_preprocessing/README.md) are currently placeholders.

## Training and Evaluation

Use `shimmer_r4` for Shimmer or `pads_r4` for PADS. Training seeds are **42, 43, and 44**; hyperparameters are in [configs/paper.json](configs/paper.json).

```bash
# Train
python run_wearable.py train --profile shimmer_r4 --seed 42 \
  --output outputs/shimmer_s42

# Evaluate
python run_wearable.py eval --profile shimmer_r4 --seed 42 \
  --checkpoint outputs/shimmer_s42/best.pt --output outputs/shimmer_s42_test
```

After the checkpoint bundle is released, omit `--checkpoint` to evaluate the pretrained model. See [PROTOCOL.md](docs/PROTOCOL.md) for reproduction details.

## Acknowledgements

We thank [Medformer](https://github.com/DL4mHealth/Medformer), [TimesNet](https://github.com/thuml/Time-Series-Library), and [TeCh](https://github.com/Levi-Ackman/TeCh) for sharing their code.
