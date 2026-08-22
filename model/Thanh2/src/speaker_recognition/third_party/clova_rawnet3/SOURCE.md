# Clova RawNet3 Source Provenance

The code in `rawnet3.py` and `blocks.py` is adapted from the official Clova
VoxCeleb trainer repository under its MIT license.

- Repository: `https://github.com/clovaai/voxceleb_trainer`
- Pinned revision: `f51bab870672a9b0b50fa158b4e30f329e7866d7`
- Upstream `models/RawNet3.py` SHA-256:
  `6d8e752e166b5e806a5afc919345f38e9353a0318ca35cdddd758c78d7bf50ef`
- Upstream `models/RawNetBasicBlock.py` SHA-256:
  `baafa9b1aa88bb249e18f94cab895ede106b768f6fff417b0189acac959a0910`
- Upstream `configs/RawNet3_AAM.yaml` SHA-256:
  `5681f648d9c5415fe61eb97e911cfffedcc1984f02af526fa437070cee1aecfc`
- Upstream `LICENSE.md` SHA-256:
  `a0e173445bf6b7717f8f7225f3d18be719e858884a20181662ffc37d8f546358`

Adaptations are limited to package-relative imports, type annotations,
docstrings, clearer local names, formatting, explicit input validation, and
removing one constructor console print. The layer topology, parameters,
checkpoint keys, and forward mathematical operations are unchanged. Strict
checkpoint loading and the audited parameter count are verified on Kaggle.
