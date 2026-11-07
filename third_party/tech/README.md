# TeCh baseline

Source: [Levi-Ackman/TeCh](https://github.com/Levi-Ackman/TeCh), revision `9a378cc546a5d97c871eff282148175b3c7cd75b`.

No LICENSE/COPYING file was found in this pinned revision. The source package therefore excludes its three implementation files and does not assign them an MIT license. To use the baseline, run:

```bash
python scripts/fetch_tech.py
```

The script downloads the pinned sources and verifies their original and import-adapted SHA256 hashes against `third_party/VENDOR_MANIFEST.json`. Follow the upstream usage terms.

`models/TeCh.py` adapts Medformer's four-argument interface to TeCh's `forward(x)`.
