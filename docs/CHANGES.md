# Changes from Medformer

Upstream revision: `891f65b8a7e77188508fd8c56e10ccdba7fc1e18`. All 49 upstream Python files are preserved byte for byte, including `run.py`, the data factory, `Exp_Classification`, models, layers, and utilities.

The root README describes PDSigformer; the original text is retained in `README_MEDFORMER.md` with an archive notice. `.gitignore` excludes local artifacts. The MIT license retains the upstream attribution and the existing 2026 SI-ML attribution. `.gitattributes` disables line-ending conversion to preserve source hashes.

Seven unused upstream notebooks were removed: `experiments.ipynb`, `supplement_experiments.ipynb`, and the five notebooks under `data_preprocessing/`. Their original hashes remain in the upstream manifest, and the source audit records these omissions explicitly. The PADS/Shimmer preprocessing files contain documentation only; training and evaluation currently use frozen prepared arrays.

| Added files / directories | Purpose |
|---|---|
| `models/PDSigformer.py`, `pds_components.py`, `pds_timesnet.py` | Residual model and frontend used by the archived experiments |
| `models/TimesNet.py`, `models/TeCh.py` | Baseline interfaces; TeCh retains the `core.` checkpoint prefix |
| `third_party/medformer_paper/` | Historical baseline architecture with adapted import namespaces |
| `third_party/timesnet/` | Pinned TimesNet dependencies, sources, and license |
| `data_provider/wearable_loader.py` | Frozen cache loading with the upstream collate function and unchanged arrays/masks |
| `exp/exp_wearable.py` | Classification subclass and historical training protocol; training never reads Test |
| `utils/pds_checkpoint.py`, `pds_runtime.py` | Hashes, metrics, random states, non-tensor state, and historical runtimes |
| `run_wearable.py`, `configs/paper.json` | Explicit training seeds and paper configurations |
| `scripts/`, `reproduction/`, `docs/` | Execution helpers, validation records, manifests, and documentation |

Reformer's compatibility adapter is bound to the corresponding classes in the isolated namespace, with strict layer and activation checks. Its formulas, PTX, and model parameters are unchanged. The wearable loader preserves PADS/Shimmer cache identities without changing the upstream data factory. The original entry point still supports prepared upstream datasets; their preprocessing notebooks remain available from Medformer.

Run `python scripts/audit_source.py` to verify the upstream files. Clones without the original Medformer commit use the bundled hash manifest. Only the seven listed notebook omissions are allowed; other missing files fail the audit. This helper is not used in training or inference.

Maintained comments, docstrings, CLI help, and usage documentation use English. Third-party Python files and SHA-locked historical operator snapshots retain their original bytes; English explanations are provided in the adjacent README files. Comment changes refresh current-file hashes in `third_party/SOURCES.json` without altering archived source hashes. Historical validation records are retained as evidence; see [RELEASE_VALIDATION.md](../reproduction/RELEASE_VALIDATION.md).
