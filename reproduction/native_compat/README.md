# Historical numerical compatibility

The three Python modules retain the exact bytes used in the original unified inference checks. Their SHA256 hashes are compared with the archived manifest before loading. Runtime binaries are imported from `artifacts/` and are not distributed with source.

- **TimesNet / Autoformer:** cuFFT 10.9 compatibility for the recorded Test batch sizes and shapes. The adapters preserve the original ATen layout and normalization behavior.
- **Reformer:** GELU compiled with CUDA 11.8. Kernel source, PTX, and the upstream PyTorch license are retained. The validated GPU architecture is SM89.
- These operators are not used for training backpropagation and do not replace model predictions with historical probabilities.

Reformer's class-namespace adaptation is implemented in `utils/pds_runtime.py`. The frozen operator modules remain unchanged, including their original Chinese comments and messages, because editing even a comment would invalidate the archived hashes. The adapters check model state, sample identities, input shapes, runtime versions, and expected call counts; Reformer also verifies that the original functions and model state are restored on exit.
