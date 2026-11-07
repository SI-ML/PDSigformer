# Third-party sources

Dependencies are pinned to the paper experiments without overwriting the upstream Medformer main files.

| Source | Pinned revision | Purpose |
|---|---|---|
| [DL4mHealth/Medformer](https://github.com/DL4mHealth/Medformer) | `446275f27b713a9f09917a6ba0bc51a18e921597` | Historical baseline models and layers; import namespaces adapted |
| [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `4e938a1767106324dd753b2a44832bf870a0252e` | TimesNet baseline and dependencies |
| [Levi-Ackman/TeCh](https://github.com/Levi-Ackman/TeCh) | `9a378cc546a5d97c871eff282148175b3c7cd75b` | Optional baseline installed with the download script |

`models/Medformer.py` remains at main revision `891f65b8a7e77188508fd8c56e10ccdba7fc1e18` and serves as the PDSigformer backbone. The historical Medformer baseline uses the paper revision with its original SWA configuration. Their checkpoints are not interchangeable.

See `SOURCES.json` and `VENDOR_MANIFEST.json` for provenance and file hashes. Upstream licenses are retained in the corresponding directories; the root license preserves Medformer's attribution. Vendored Python files retain their original comments and docstrings to preserve recorded hashes.
