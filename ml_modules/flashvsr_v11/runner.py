"""Execute the unchanged official v1.1 Tiny Long example."""

import argparse
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Keep direct CLI execution independent of the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml_modules.flashvsr_v11.runtime import MODEL_FILES, SOURCE_REVISION


def load_official(cache):
    source = cache / "upstream"
    if not source.is_dir():
        raise RuntimeError(f"Missing official source at {source}. Run python -m ml_modules.flashvsr_v11.setup first.")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=30)
    if revision != SOURCE_REVISION or dirty:
        raise RuntimeError(f"FlashVSR requires unchanged official source at {SOURCE_REVISION}")
    examples = source / "examples/WanVSR"
    sys.path[:0] = [str(source), str(examples)]
    try:
        import torch  # Load libtorch before importing the native attention extension.
        import block_sparse_attn  # Required by the official LCSA implementation.
        spec = importlib.util.spec_from_file_location("official_flashvsr_v11", examples / "infer_flashvsr_v1.1_tiny_long_video.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError("FlashVSR v1.1 runtime dependencies are missing/incompatible; see ml_modules/flashvsr_v11/README.md.") from exc
    return module, examples


def move_cached_context(pipe, device):
    for block in pipe.denoising_model().blocks:
        attention = block.cross_attn
        attention.cache_k = attention.cache_k.to(device)
        attention.cache_v = attention.cache_v.to(device)
    pipe.prompt_emb_posi["context"] = pipe.prompt_emb_posi["context"].to(device)
    for name in ("timestep", "t", "t_mod"):
        setattr(pipe, name, getattr(pipe, name).to(device))


def main(preloaded=None, model_cache=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--fast-inference", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", default="24")
    parser.add_argument("--content-frames", type=int)
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    args = parser.parse_args()
    cache = args.cache.resolve()
    stage_start = time.perf_counter()
    official, examples = preloaded if preloaded is not None else load_official(cache)
    print(f"FLASHVSR_STAGE imports_and_source_check={time.perf_counter() - stage_start:.6f}", flush=True)
    if args.check_imports:
        print("Official FlashVSR v1.1 imports OK")
        return
    fast_runtime = None
    if args.fast_inference:
        from ml_modules.flashvsr_v11.fast_inference import check_fast_dependencies
        fast_runtime = check_fast_dependencies(official.torch)
    for filename in MODEL_FILES:
        if not (cache / "models" / filename).is_file():
            raise RuntimeError(f"Missing {cache / 'models' / filename}. Run python -m ml_modules.flashvsr_v11.setup first.")
    torch = official.torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Official FlashVSR v1.1 requires a CUDA GPU with BF16 support")
    if args.check:
        if fast_runtime is not None:
            print("Fast inference dependencies:", json.dumps(fast_runtime, sort_keys=True))
        print("Official FlashVSR v1.1 runtime/assets OK")
        return
    if args.input is None or args.output is None or args.work_dir is None:
        parser.error("inference requires --input, --output and --work-dir")
    input_path, output_path = args.input.resolve(), args.output.resolve()
    # Reproduce the example's relative asset paths in job-local staging without
    # editing upstream files or writing symlinks into the shared source checkout.
    cwd = args.work_dir.resolve() / "examples/WanVSR"
    cwd.mkdir(parents=True)
    (cwd / "prompt_tensor").symlink_to(examples / "prompt_tensor", target_is_directory=True)
    (cwd / "FlashVSR-v1.1").symlink_to(cache / "models", target_is_directory=True)
    os.chdir(cwd)

    def make_fast_input():
        from ml_modules.flashvsr_v11.fast_inference.input import ChunkedLQ
        content_frames = args.content_frames
        if content_frames is None:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
                 str(input_path)],
                capture_output=True, text=True, check=True, timeout=600,
            )
            content_frames = int(probe.stdout.strip())
        source_size = None
        if args.input_width is not None or args.input_height is not None:
            if args.input_width is None or args.input_height is None:
                raise ValueError("Both staged input dimensions are required")
            source_size = (args.input_width, args.input_height)
        return ChunkedLQ(input_path, content_frames, content_frames + 4,
                         dtype=torch.bfloat16, source_size=source_size), content_frames

    with torch.no_grad():
        reuse_model = model_cache is not None and model_cache.get("fast") == args.fast_inference
        early_input = args.fast_inference and not reuse_model
        if early_input:
            stage_start = time.perf_counter()
            lq, content_frames = make_fast_input()
            print(f"FLASHVSR_STAGE early_input={time.perf_counter() - stage_start:.6f}", flush=True)
        stage_start = time.perf_counter()
        if reuse_model:
            pipe = model_cache["pipe"].to("cuda")
            pipe.load_models_to_device(["dit", "vae"])
            move_cached_context(pipe, "cuda")
        else:
            pipe = official.init_pipeline()
        print(f"FLASHVSR_STAGE model_init={time.perf_counter() - stage_start:.6f}", flush=True)
        if args.fast_inference and not reuse_model:
            stage_start = time.perf_counter()
            from ml_modules.flashvsr_v11.fast_inference import install_fast_inference
            configuration = install_fast_inference(pipe)
            print("FAST_INFERENCE_CONFIGURATION " + json.dumps(configuration, sort_keys=True), flush=True)
            print(f"FLASHVSR_STAGE fast_setup={time.perf_counter() - stage_start:.6f}", flush=True)
        if model_cache is not None:
            model_cache.update(pipe=pipe, fast=args.fast_inference)
        stage_start = time.perf_counter()
        if args.fast_inference and not early_input:
            lq, content_frames = make_fast_input()
        if args.fast_inference:
            height, width, frames = lq.height, lq.width, lq.total_frames
        else:
            lq, height, width, frames, _ = official.prepare_input_tensor(
                str(input_path), scale=4.0, dtype=torch.bfloat16, device="cuda",
            )
        # Long LTX clips can exceed the model's default 1024 temporal positions.
        # Recompute the cache with the official helper; existing positions and
        # the model's positional encoding are numerically unchanged.
        dit = pipe.denoising_model()
        latent_frames = (frames - 1) // 4
        if latent_frames > len(dit.freqs[0]):
            from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d
            dit.freqs = precompute_freqs_cis_3d(dit.blocks[0].self_attn.head_dim, end=latent_frames)
        print(f"FLASHVSR_STAGE input_setup={time.perf_counter() - stage_start:.6f}", flush=True)
        stage_start = time.perf_counter()
        try:
            if args.fast_inference:
                if (frames - 1) % 8 != 0:
                    raise ValueError(f"Fast input must be padded to the 8n-3 grid before lookahead, got {content_frames} frames")
                # Prime the first model step while retaining a bounded producer queue.
                lq[:, :, :min(21, frames), :, :]
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=args.seed,
                LQ_video=lq, num_frames=frames, height=height, width=width,
                is_full_block=False, if_buffer=True, topk_ratio=2.0 * 768 * 1280 / (height * width),
                kv_ratio=3.0, local_range=11, color_fix=True,
            )
        finally:
            if args.fast_inference:
                lq.close()
                print(f"FAST_INPUT_STATS {json.dumps({'queue_capacity': lq.queue.maxsize, 'producer_seconds': lq.producer_seconds, 'consumer_wait_seconds': lq.wait_seconds})}", flush=True)
        print(f"FLASHVSR_STAGE inference={time.perf_counter() - stage_start:.6f}", flush=True)
        # Write at the source's rational FPS; the official demo rounds it to an
        # integer. Inference itself is unchanged. Frame conversion is per-frame
        # to avoid an extra full-video float/NumPy copy on CPU.
        command = [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(Fraction(args.fps)), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        stage_start = time.perf_counter()
        convert_seconds = 0.0
        write_seconds = 0.0
        try:
            for frame in video.permute(1, 2, 3, 0):
                convert_start = time.perf_counter()
                pixels = ((frame.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                convert_seconds += time.perf_counter() - convert_start
                write_start = time.perf_counter()
                encoder.stdin.write(pixels.tobytes())
                write_seconds += time.perf_counter() - write_start
            encoder.stdin.close()
            wait_start = time.perf_counter()
            if encoder.wait(timeout=600):
                raise RuntimeError("FlashVSR video encoding failed")
            print(f"FLASHVSR_STAGE encode_convert={convert_seconds:.6f} encode_write={write_seconds:.6f} encode_wait={time.perf_counter() - wait_start:.6f}", flush=True)
            print(f"FLASHVSR_STAGE encode={time.perf_counter() - stage_start:.6f}", flush=True)
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait(timeout=30)


if __name__ == "__main__":
    main()
