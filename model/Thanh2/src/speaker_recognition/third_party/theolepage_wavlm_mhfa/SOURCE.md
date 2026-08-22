# WavLM+MHFA Source Provenance

The MHFA code in `mhfa.py` is adapted from the official project under its MIT
license. The larger WavLM implementation is not duplicated here. At runtime,
the adapter downloads the exact pinned source snapshot, verifies both required
Python files by SHA-256, and only then imports them.

- Repository: `https://huggingface.co/theolepage/wavlm_ssl_sv`
- Pinned revision: `bfb8527de83b5347fb81b1e9e31be241656ca103`
- Upstream `models/Baseline/Spk_Encoder.py` SHA-256:
  `fc4638d657a3ad09953e54ee76ae4877904b6a92c03d2f91ed6691b7f770d40f`
- Upstream `models/Baseline/WavLM.py` SHA-256:
  `7cc0837302ff032d048c0f43ebdafdf0f009f72a10d78aa82f486df33c39aa63`
- Upstream `models/Baseline/modules.py` SHA-256:
  `7a06a14a7dc95c5f65cd6b09ed126013821512489dcfec2e58bd8b544ce46656`
- Upstream `LICENSE.md` SHA-256:
  `04a05562ba5e9841452b4e7209d226e543b975d0809794452c9e301f457a183a`

These are canonical Git-blob byte hashes with LF line endings, matching files
resolved by Hugging Face on Linux. They are deliberately not hashes of a
Windows checkout after `core.autocrlf` conversion.

MHFA adaptations are limited to type annotations, docstrings, formatting,
explicit shape validation, and clearer local names. Parameter-bearing
attribute names, parameter shapes, and forward mathematical operations are
unchanged. Strict checkpoint loading verifies compatibility on Kaggle.
