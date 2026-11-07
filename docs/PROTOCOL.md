# Fixed paper protocol

The main comparison includes 13 baselines and PDSigformer. Ablations use R1-R4. Both datasets use three training seeds and the fixed subject-independent `split42` partition.

| Configuration | TimesNet and residual fusion | Multiple granularities | Inter-granularity attention |
|---|---|---|---|
| R1 | No | No | No |
| R2 | Yes | No | No |
| R3 | Yes | Yes | No |
| R4 / PDSigformer | Yes | Yes | Yes |

All configurations retain intra-granularity attention. R1/R2 use a single patch length: `2` for Shimmer and `32` for PADS. R3/R4 granularities are specified in [configs/paper.json](../configs/paper.json). R1 is initialized by constructing R2 and extracting its backbone; R3 follows R4 construction before removing inter-granularity attention.

PDSigformer's backbone uses Medformer main `891f65b...`; historical Medformer and native baselines use paper revision `446275f...` in an isolated namespace. FEDformer's non-tensor frequency indices are saved and restored as `construction_metadata`. Checkpoints require exact key matches. Historical PDSigformer inference also verifies the initialization hash, including fixed positional-encoding buffers.

## Data and metrics

- Labels: HC=0 and PD=1. Model input: `[B,T,6]`.
- Apply `StandardScaler` along time for each window and channel, including existing padded values.
- Prepared caches are not normalized, cropped, windowed, or padded again. Preserve evaluation batch size, ordering, and all-one masks.
- Accuracy uses argmax. Precision, recall, and F1 use macro averaging. AUROC and AUPRC use two-class one-vs-rest macro averaging; AUPRC is computed with `average_precision_score`.
- Report means and sample standard deviations (`ddof=1`) over three seeds. Repeated inference checks are not additional seeds.

## Training

PDSigformer uses Adam; the TimesNet baseline uses RAdam; other baselines use Adam. Training uses `weight_decay=0`, a constant learning rate, unweighted cross-entropy, and gradient clipping at 4. The Medformer baseline updates SWA each epoch and uses its validation score for checkpoint selection. PDSigformer does not use SWA.

The checkpoint improvement threshold is `delta=1e-5`; the epoch budget and patience are both 100. Evaluation uses the validation-selected checkpoint, not necessarily the last epoch. Baseline evaluation seeds are derived from seed/model/task/split, with the training random state restored afterward. Historical PDSigformer Test follows: set seed, construct the model, load weights, then evaluate.

Training loads only Train/Valid. Test runs through a separate `eval` command. The wearable entry point reuses the upstream classification base class, model interface, and loss, while implementing the archived training, SWA, checkpoint-selection, and logging protocol.

## Validation scope

Inference reproduction requires matching weights, architecture, data, ordering, batch size, and runtimes. FFT/GELU compatibility operators are used only for historical Test runs; training uses standard differentiable PyTorch operators. Environment changes may alter training trajectories. Short training checks do not establish that full 100-epoch retraining will recover the reported paper scores.
