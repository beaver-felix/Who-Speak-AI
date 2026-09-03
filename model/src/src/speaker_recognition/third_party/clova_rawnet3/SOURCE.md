# Clova RawNet3 Source Provenance

The code in `rawnet3.py` and `blocks.py` is adapted from the official Clova
VoxCeleb trainer repository under its MIT license.

- Repository: `https://github.com/clovaai/voxceleb_trainer`
- Pinned revision: `f51bab870672a9b0b50fa158b4e30f329e7866d7`
- Upstream `models/RawNet3.py` SHA-256:
  `8daf5e486055fceda56b0571fc70c685bf9d19a6f41463f4193b88a1f34b1636`
- Upstream `models/RawNetBasicBlock.py` SHA-256:
  `63795b5d0cbde5b1cd502fac09ea84962184abd96446e65851b3d400c2454f76`
- Upstream `configs/RawNet3_AAM.yaml` SHA-256:
  `86147c93a19287fef53accd4137d6474190db78c6833e8d9dc4f196331e3d5f8`
- Upstream `LICENSE.md` SHA-256:
  `c8782ff21c6915be5f1edadd5f2515cfb1ea07e092338fcff5e1ae569e7f17f8`

These are canonical Git-blob byte hashes with LF line endings rather than
hashes after a Windows `core.autocrlf` checkout conversion.

Adaptations are limited to package-relative imports, type annotations,
docstrings, clearer local names, formatting, explicit input validation, and
removing one constructor console print. The layer topology, parameters,
checkpoint keys, and forward mathematical operations are unchanged. Strict
checkpoint loading and the audited parameter count are verified on Kaggle.
