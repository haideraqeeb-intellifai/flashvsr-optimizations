# PR #1715 video adapter and preprocessing optimizations

This directory is a public snapshot of the FlashVSR v1.1 adapter at
`strontiumcorp/script2video` PR #1715, head `f9a58c5ba202a1c72765ea7860ecfd51b7e35b0e`.
It contains the model adapter and fast-inference modules; the private LTX worker
is not included. The public `run_pr1715_video.py` entry point can enhance a local
video using this adapter. This path is separate from the repository's original
six-clip study and uses its own Python 3.10 / Torch 2.10 environment.

The preprocessing changes in this snapshot are:

1. Probe H.264 MP4 packet counts and compare them with container frame counts;
   decode-count other formats when needed.
2. Resize, pad and frame-extend to FlashVSR's 4× / 8n−3 input grid using lossless
   x264 with the `ultrafast` preset. The staged decoded pixels match the earlier
   lossless `fast` preset.
3. Pass known staged frame count and dimensions to the runner so it skips a
   second full-video count and an initial frame decode.
4. Start the bounded, ordered 32-frame input producer before first model setup;
   overlap PIL bicubic preparation with model work.
5. Reuse official imports and a configured CPU model across requests in the
   parent process. Retain CPU master tensor references so later GPU releases
   restore those references without copying weights back from GPU. Clear
   transient CUDA state after each request.

The optimized attention, NVFP4 FFNs, FP8 decoder and LQ modules are included so
the adapter is inspectable as a complete unit. The baseline BF16 path remains
available by omitting `--fast-inference`.

## Runtime and assets

The measured runtime used Python 3.10.19, Torch 2.10.0+cu130, Transformers
4.57.6, FlashInfer 0.7.0.post1, cuDNN frontend 1.31.0, TVM FFI 0.1.14.post1,
and a Block-Sparse-Attention 0.0.2 wheel built from revision
`49d6c39e4dc0303442cda3bb758b3925d4399c49` for SM120. It requires an
RTX 5090, CUDA 13.1 toolkit, Ninja, FFmpeg and FFprobe. The existing root
`setup.sh` creates a different Torch 2.14 study environment and is not the
measured adapter environment. See the [PR environment recipe](https://github.com/strontiumcorp/script2video/blob/f9a58c5ba202a1c72765ea7860ecfd51b7e35b0e/envs/env_8_flashtalk_5090_uv.sh)
for its exact package installation, or build a compatible environment from the
versions above. The pinned model and attention references are in the root
`MODEL_REFERENCES.md` and `provenance/assets.json`; no weights are in Git.

From a compatible environment at the repository root:

```bash
python -m ml_modules.flashvsr_v11.setup
python -m ml_modules.flashvsr_v11.runner --cache "$PWD/cache/flashvsr_v11" --check --fast-inference
python run_pr1715_video.py input.mp4 output.mp4 --width 1280 --height 736 --seed 7 --fast-inference
```

The setup command fetches official source revision
`cf910c61a60733e610e9c6e8b607f80c3a6c202b` and model revision
`27561b186ded3402d7c975f4fd722e2885b6135f` into the ignored cache.
The CLI copies its input and leaves it intact. It outputs the original frame
count and rational FPS, requested dimensions and copied audio streams.

## Measured 736p requests

`benchmarks/pr1715_20261022.json` records the individual local-copy-upload
request times, prompt and seed. The 241-frame, 24-fps outputs all passed
1280×736 metadata checks. These measurements exclude asset download,
environment installation, queue time and callbacks. The new adapter's warm
optimized request took 70.19s versus 79.00s in the earlier same-host run;
its incremental SR overhead was 13.08s versus 22.61s. These numbers are
single requests, not averages or a quality comparison.

The adapter's parent-process CUDA inference does not have a separate child
timeout. A caller should enforce an outer job deadline. The CPU model cache
retains several gigabytes of host RAM between requests.
