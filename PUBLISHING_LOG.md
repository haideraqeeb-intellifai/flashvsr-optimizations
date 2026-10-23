# Publishing log

## 2026-10-23 17:50 IST

Added the PR #1715 FlashVSR adapter source snapshot, a local-video entry point,
and the 1280×736 benchmark inputs and individual timings. The adapter source
matches PR head `f9a58c5ba202a1c72765ea7860ecfd51b7e35b0e` apart from
public-path documentation and removal of an unused internal model-store prefix.
The prior six-clip study remains intact. Excluded: the private LTX worker,
model weights, generated videos, caches and installed environments.

## 2026-10-14 03:51 IST

Initial public code publication to `https://github.com/haideraqeeb-intellifai/flashvsr-optimizations` on `main`. Source directory: `/home/raqeeb/fastvsr-optimizations`, which has no usable Git history. Pinned upstream source revisions are in `provenance/assets.json`.

Inventory: benchmark runner, chunked input and exact GPU resize modules, attention/FFN/decoder/LQ optimization adapters, setup and validation scripts, dependency files, the calibrated head-retention policy, the public source input manifest, and model references.

Excluded: private local measurements and profiles; model weights; generated and reference videos; installed dependencies, caches, and fetched upstream checkouts. Their reproduction or retrieval paths are described in `README.md` and `MODEL_REFERENCES.md`.
