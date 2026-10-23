# FlashVSR v1.1 optimization and chunked input benchmarks

## PR #1715 preprocessing and parent-process adapter

The [PR #1715 adapter](ml_modules/flashvsr_v11/README.md) is now included as a
separate, inspectable implementation. It adds lossless ultrafast staging,
passes known frame count and dimensions to the input queue, overlaps initial
frame preparation with model setup, and reuses a configured FlashVSR model in
the parent process with CPU master weights. A local-video entry point is
[`run_pr1715_video.py`](run_pr1715_video.py). Its exact request inputs and
single-run timings are in [`benchmarks/pr1715_20261022.json`](benchmarks/pr1715_20261022.json).

The new adapter uses the PR's Python 3.10 / Torch 2.10 stack; the original
six-clip study below retains its Python 3.12 / Torch 2.14 setup and results.

Code for the six-clip RTX 5090 FlashVSR v1.1 Tiny Long study. The combined optimization uses layer/head BF16 sparse attention, NVFP4 transformer FFNs, an FP8 decoder, BF16 Conv3D layout/autotuning, and FP8 LQ Conv3D 2. The original local study averaged **29.7018 s of synchronized inference** across six clips. The subsequent input study overlaps preparation with inference through a bounded 32-frame producer queue. Its pixel-exact GPU path averaged **44.8566 s per finished video**, versus **53.1887 s** for the earlier eager control.

These measurements are historical single warmed passes per clip. Chunked model-phase times include overlapping input preparation and cannot be treated as isolated model timings.

## Key code

| Component | Files |
| --- | --- |
| Runner and input modes | `benchmark.py`, `chunked_input.py`, `pil_exact_gpu.py` |
| Model changes | `sparse_study_backends.py`, `ffn_quantization.py`, `decoder_optimizations.py`, `decoder_fp8.py`, `lq_optimizations.py`, `lq_fp8.py` |
| Reproduction and validation | `setup.sh`, `fetch_assets.py`, `run_lq_study.sh`, `validate_lq.py`, `profile_components.py` |
| Calibrated runtime configuration | `config/head-policy.json` |

The pinned upstream FlashVSR and attention repositories are fetched by `setup.sh`; their source code and licenses remain in their respective repositories. Model files are fetched by `fetch_assets.py` and never stored in this repository. See [MODEL_REFERENCES.md](MODEL_REFERENCES.md).

## Reproduce

Use an NVIDIA RTX 5090, Python 3.12, CUDA 13.1 toolkit, FFmpeg/FFprobe, Git, and `uv`. The Python setup pins PyTorch 2.14.0 with CUDA 13.0 runtime and builds the official block-sparse attention extension for SM120.

```bash
bash setup.sh
.venv/bin/python validate_lq.py
.venv/bin/python -u benchmark.py \
  --attention headwise_bf16 --ffn nvfp4 --decoder fp8 --lq combined \
  --warmup --output-dir outputs/reproduction-eager
.venv/bin/python -u benchmark.py \
  --attention headwise_bf16 --ffn nvfp4 --decoder fp8 --lq combined \
  --input-mode chunked_gpu_exact --warmup \
  --output-dir outputs/reproduction-gpu-exact
```

`setup.sh` downloads the six inputs and reference outputs using the URLs and SHA-256 hashes in `provenance/benchmark_manifest.json`, and downloads the pinned weights using `provenance/assets.json`. It does not send inference to an API. Use a new output directory for each run. `bash run_lq_study.sh` runs the four LQ configurations and their component profiles.

The publication copy makes two setup changes outside the timed measurements: `benchmark.py` generates its own warmup source video, and `profile_components.py` uses the LQ study baseline rather than an earlier workspace baseline. `setup.sh` installs the FFN, decoder, and analysis requirements used by these runs. The recorded results remain from the original study; new results may vary with hardware and software state.

## Provenance and outputs

- The benchmark writes new per-clip results to the requested output directory. `run_lq_study.sh` writes its measurements and profiles under `outputs/lq-study/`.
- Raw measurements and profiles from the original local study are not included in this public code repository.
- `provenance/assets.json` pins upstream source and model revisions. `provenance/attention-torch214-cxx20.patch` records the build compatibility patch.

Generated and reference MP4s are not included in this code publication. The benchmark input and reference URLs are recorded in `provenance/benchmark_manifest.json`; `fetch_assets.py` retrieves them with checksum verification. New benchmark runs generate their own output videos. The local `.venv`, caches, vendor checkouts, and model weights are also excluded because setup recreates them from pinned references.

No license is asserted for this repository's original code. Upstream code and model licenses are governed by their respective projects.
