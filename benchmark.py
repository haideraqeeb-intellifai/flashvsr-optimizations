"""Direct, local reproduction of the six-clip FlashVSR v1.1 benchmark.

No inference service or worker API is used. The pinned upstream model and
preprocessor are imported unchanged; this file owns media staging and timing.
"""
from __future__ import annotations

import argparse
import contextlib
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def command(args):
    return subprocess.check_output([str(a) for a in args], text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path, count=False):
    args = ["ffprobe", "-v", "error"]
    if count:
        args += ["-count_frames"]
    return json.loads(command(args + ["-show_streams", "-show_format", "-of", "json", path]))


def video_stream(metadata):
    return next(s for s in metadata["streams"] if s["codec_type"] == "video")


def padded_frames(frames):
    # The upstream example adds four look-ahead frames, then rounds down to
    # 8n+1. Supply 8n-3 content frames first so no real frames are discarded.
    return max(21, frames + (5 - frames) % 8)


def audio_packets(path):
    data = json.loads(command([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_packets",
        "-show_data_hash", "sha256", "-show_entries",
        "packet=stream_index,pts_time,dts_time,duration_time,size,data_hash",
        "-of", "json", path,
    ]))
    return data.get("packets", [])


def validate(source, output, expected_frames, fps):
    v = video_stream(probe(output, count=True))
    source_audio, output_audio = audio_packets(source), audio_packets(output)
    checks = {
        "resolution": (v["width"], v["height"]) == (1664, 896),
        "frame_count": int(v["nb_read_frames"]) == expected_frames,
        "fps": Fraction(v["avg_frame_rate"]) == fps,
        "duration": abs(float(v["duration"]) - expected_frames / float(fps)) < 1e-5,
        "audio_identical": bool(source_audio) and source_audio == output_audio,
    }
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output),
                    "-f", "null", "-"], check=True)
    checks["full_decode"] = True
    if not all(checks.values()):
        raise RuntimeError(f"Output integrity failure: {checks}")
    return checks


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/baseline")
    parser.add_argument("--cases", nargs="*", help="Default: all six cases in report order")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--attention", choices=['baseline','sparse_sage','sage22_dense','sage3_dense','jev_sparse','jev_replay','sparge2_10','headwise_bf16','headwise_sparge2','triton_int8_fp8_10','jev_replay_bf16','jev_replay_sparge2'], default='baseline')
    parser.add_argument("--ffn", choices=['bf16','fp8','nvfp4'], default='bf16')
    from decoder_optimizations import MODES
    parser.add_argument('--decoder', choices=MODES, default='baseline')
    from lq_optimizations import MODES as LQ_MODES
    parser.add_argument('--lq', choices=LQ_MODES, default='baseline')
    parser.add_argument("--warmup", action='store_true', help='Untimed 49-frame pipeline warmup before measured clips')
    parser.add_argument('--input-mode', choices=['eager', 'chunked', 'chunked_gpu', 'chunked_gpu_exact'], default='eager',
                        help='Eager PIL, overlapping PIL, GPU bicubic, or GPU bicubic with exact Pillow arithmetic')
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / "provenance/benchmark_manifest.json").read_text())
    assets = json.loads((ROOT / "provenance/assets.json").read_text())
    cases = [c for c in report["cases"] if not args.cases or c["case"] in args.cases]
    if not cases or (args.cases and set(args.cases) - {c["case"] for c in cases}):
        parser.error("Unknown or empty case selection")
    if any((output_dir / c["case"] / "enhanced.mp4").exists() for c in cases):
        raise FileExistsError("Choose a new output directory to preserve previous measurements")

    # Verify immutable inputs and model assets outside the timed processing.
    source = ROOT / "vendor/FlashVSR"
    revision = command(["git", "-C", source, "rev-parse", "HEAD"])
    assert revision == assets["source_revision"], revision
    assert not command(["git", "-C", source, "status", "--porcelain", "--untracked-files=no"])
    attention_source = ROOT / "vendor/Block-Sparse-Attention"
    assert command(["git", "-C", attention_source, "rev-parse", "HEAD"]) == assets["attention_revision"]
    attention_patch = (ROOT / "provenance/attention-torch214-cxx20.patch").read_text().strip()
    assert command(["git", "-C", attention_source, "diff", "HEAD"]) == attention_patch
    for name, expected in assets["model_sha256"].items():
        assert sha256(ROOT / "models/FlashVSR-v1.1" / name) == expected, name
    for case in cases:
        path = ROOT / "inputs" / case["input"]["url"].split("/")[-1]
        assert sha256(path) == case["input"]["sha256"], path

    examples = source / "examples/WanVSR"
    sys.path[:0] = [str(source), str(examples)]
    import torch
    import block_sparse_attn
    import block_sparse_attn_cuda
    torch.set_num_threads(args.threads)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.__version__ == report["flashvsr_v11"]["torch"], torch.__version__
    spec = importlib.util.spec_from_file_location(
        "official_flashvsr", examples / "infer_flashvsr_v1.1_tiny_long_video.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    # Upstream resolves its prompt as ../../examples/WanVSR/prompt_tensor.
    runtime = output_dir / "runtime/examples/WanVSR"
    runtime.mkdir(parents=True, exist_ok=True)
    for name, target in [("FlashVSR-v1.1", ROOT / "models/FlashVSR-v1.1"),
                         ("prompt_tensor", examples / "prompt_tensor")]:
        if not (runtime / name).exists():
            (runtime / name).symlink_to(target, target_is_directory=True)
    os.chdir(runtime)
    from diffsynth.models import wan_video_dit
    from attention_backends import AttentionBackend
    if args.attention in ('sparge2_10','headwise_bf16','headwise_sparge2','triton_int8_fp8_10','jev_replay_bf16','jev_replay_sparge2'):
        from sparse_study_backends import SparseStudyBackend
        backend = SparseStudyBackend(args.attention)
    else:
        backend = AttentionBackend(args.attention)
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads(),
        "ffmpeg": command(["ffmpeg", "-version"]).splitlines()[0],
        "nvidia_smi": command(["nvidia-smi"]),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "attention_module": block_sparse_attn.__file__,
        "attention_extension_sha256": sha256(block_sparse_attn_cuda.__file__),
        "attention_build_patch": attention_patch,
        "runner_sha256": sha256(Path(__file__)),
        "dense_attention": {key: getattr(wan_video_dit, key) for key in
                            ["FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"]},
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "assets": assets,
        "attention_backend": backend.metadata(),
    }
    write_json(output_dir / "environment.json", environment)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        pipe = official.init_pipeline()
    torch.cuda.synchronize()
    startup = time.perf_counter() - started
    print(f"MODEL_STARTUP_SECONDS {startup:.6f}", flush=True)
    from ffn_quantization import install_ffn
    quant_start = time.perf_counter()
    with torch.no_grad():
        ffn_metadata = install_ffn(pipe,args.ffn)
    torch.cuda.synchronize()
    ffn_metadata['setup_seconds'] = time.perf_counter()-quant_start
    environment['ffn'] = ffn_metadata
    environment['ffn_adapter_sha256'] = sha256(ROOT/'ffn_quantization.py')
    from decoder_optimizations import install_decoder_optimization
    decoder_metadata = install_decoder_optimization(pipe, args.decoder)
    if args.decoder in ('fp8', 'nvfp4'):
        ffn_metadata['scope'] = 'Only dit.blocks.*.ffn.{0,2}; decoder precision is configured separately and recorded in decoder metadata'
        environment[f'decoder_{args.decoder}_sha256'] = sha256(ROOT / f'decoder_{args.decoder}.py')
    torch.cuda.synchronize()
    environment['decoder'] = decoder_metadata
    environment['decoder_adapter_sha256'] = sha256(ROOT/'decoder_optimizations.py')
    from lq_optimizations import install_lq
    lq_metadata, _ = install_lq(pipe, args.lq)
    environment['lq'] = lq_metadata
    environment['lq_adapter_sha256'] = sha256(ROOT/'lq_optimizations.py')
    environment['lq_fp8_sha256'] = sha256(ROOT/'lq_fp8.py')
    environment['cudnn_version'] = torch.backends.cudnn.version()
    write_json(output_dir / 'environment.json',environment)
    if args.warmup:
        if args.attention in ('jev_replay_bf16','jev_replay_sparge2'):
            backend.before_inference(next(c for c in report['cases'] if c['case']=='female_default'))
        residuals = {}
        hooks = []
        if args.attention == 'baseline':
            def observe(index):
                def hook(module, inputs, output):
                    x = inputs[0][:,:64,:16].float()
                    y = (output[0] if isinstance(output,tuple) else output)[:,:64,:16].float()
                    residuals.setdefault(str(index),[]).append(float((y-x).norm()/x.norm().clamp_min(1e-8)))
                return hook
            hooks = [b.register_forward_hook(observe(i)) for i,b in enumerate(pipe.dit.blocks)]
        # Build the warmup input in this run's output directory so a fresh
        # checkout does not depend on a previous baseline run.
        warm_case = next(c for c in report['cases'] if c['case'] == 'female_default')
        warm_input = ROOT / 'inputs' / warm_case['input']['url'].split('/')[-1]
        warm_source = output_dir / '_warmup_low_resolution_padded.mp4'
        warm_padding = padded_frames(warm_case['input']['frames']) - warm_case['input']['frames']
        subprocess.run([
            'ffmpeg', '-y', '-v', 'error', '-i', str(warm_input), '-map', '0:v:0', '-an',
            '-vf', f'scale=426:240:flags=bicubic,setsar=1,tpad=stop_mode=clone:stop={warm_padding}',
            '-frames:v', str(warm_case['input']['frames'] + warm_padding), '-r', '24',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '0', '-pix_fmt', 'yuv444p', str(warm_source),
        ], check=True)
        with torch.no_grad(), open(output_dir / 'warmup.log','w') as log, contextlib.redirect_stdout(log):
            warm_lq, wh, ww, _, _ = official.prepare_input_tensor(str(warm_source), scale=4, dtype=torch.bfloat16, device='cuda')
            warm_video = pipe(prompt='', negative_prompt='', cfg_scale=1.0, num_inference_steps=1, seed=2026,
                LQ_video=warm_lq[:,:,:49].contiguous(), num_frames=49, height=wh, width=ww,
                is_full_block=False, if_buffer=True, topk_ratio=2.0*768*1280/(wh*ww),
                kv_ratio=3.0, local_range=11, color_fix=False)
            torch.cuda.synchronize()
            assert torch.isfinite(warm_video).all(), 'Nonfinite warmup output'
            del warm_lq, warm_video
        for hook in hooks:
            hook.remove()
        if residuals:
            write_json(output_dir/'pilot-residuals.json', {k:statistics.mean(v) for k,v in residuals.items()})
        print('WARMUP_COMPLETE', flush=True)
    results = {
        "status": "running", "configuration": report["flashvsr_v11"]["settings"],
        "model_startup_seconds_shared": startup, "cases": [],
        "input_mode": args.input_mode,
        "attention_backend": backend.metadata(), "warmup_49_frames": args.warmup,
        "ffn": ffn_metadata,
        "decoder": decoder_metadata,
        "lq": lq_metadata,
        "timing_notes": {
            "processing": "LR staging + input preparation + synchronized inference + official materialization/encoding + trim/audio mux. Excludes model startup, downloads, asset hashing and integrity verification.",
            "chunked_input": "For chunked input modes, input_tensor_preparation primes 21 frames and inference includes concurrent input preparation, any queue waits, and the model/decoder. These phases cannot be added to a model-only baseline timing in isolation.",
            "report_comparable": "Also expose processing excluding LR staging because the original report does not disclose whether that stage was timed.",
            "undocumented_choices": "Report does not specify LR resize filter/codec, encoder settings, CPU threads or full dependency lock. We use bicubic, lossless H264 yuv444p staging, upstream quality=5 encoding, final libx264 fast CRF18, eight Torch CPU threads.",
            "lip_sync": "SyncNet has not been run; media integrity does not constitute a new SyncNet verdict.",
        },
    }
    for case in cases:
        case_dir = output_dir / case["case"]
        case_dir.mkdir(exist_ok=True)
        src = ROOT / "inputs" / case["input"]["url"].split("/")[-1]
        lr = case_dir / "low_resolution_padded.mp4"
        intermediate = case_dir / "official_encoded.mp4"
        dst = case_dir / "enhanced.mp4"
        metadata = video_stream(probe(src))
        n = int(metadata["nb_frames"])
        fps = Fraction(metadata["avg_frame_rate"])
        assert n == case["input"]["frames"] and fps == 24
        padding = padded_frames(n) - n
        assert padding == case["tail_padding"]["temporary_frames"]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        print(f"CASE {case['case']}: staging {n} real frames + {padding} tail frames", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an",
            "-vf", f"scale=426:240:flags=bicubic,setsar=1,tpad=stop_mode=clone:stop={padding}",
            "-frames:v", str(n + padding), "-r", str(fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", "0", "-pix_fmt", "yuv444p", str(lr),
        ], check=True)
        staged = time.perf_counter()
        if args.input_mode in ('chunked', 'chunked_gpu', 'chunked_gpu_exact'):
            from chunked_input import ChunkedLQ
            lq = ChunkedLQ(lr, n + padding, n + padding + 4,
                           resize_mode={'chunked': 'pil', 'chunked_gpu': 'gpu',
                                        'chunked_gpu_exact': 'gpu_exact'}[args.input_mode])
            # The first model step consumes 21 frames; prime just that group.
            lq[:, :, :21, :, :]
            height, width, frames = lq.height, lq.width, lq.total_frames
        else:
            with open(case_dir / "prepare.log", "w") as log, contextlib.redirect_stdout(log):
                lq, height, width, frames, _ = official.prepare_input_tensor(
                    str(lr), scale=4, dtype=torch.bfloat16, device="cuda")
        assert (height, width, frames) == (896, 1664, n + padding + 4)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        print(f"CASE {case['case']}: inference {frames} frames at {width}x{height}", flush=True)
        backend.reset()
        backend.before_inference(case)
        with torch.no_grad():
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=2026,
                LQ_video=lq, num_frames=frames, height=height, width=width,
                is_full_block=False, if_buffer=True, topk_ratio=2.0 * 768 * 1280 / (height * width),
                kv_ratio=3.0, local_range=11, color_fix=False,
            )
        torch.cuda.synchronize()
        inferred = time.perf_counter()
        input_stats = None
        if args.input_mode in ('chunked', 'chunked_gpu', 'chunked_gpu_exact'):
            lq.close()
            input_stats = {"producer_active_seconds": lq.producer_seconds,
                           "consumer_queue_wait_seconds": lq.wait_seconds,
                           "maximum_lookahead_frames": lq.queue.maxsize,
                           "resize_mode": lq.resize_mode}
        assert torch.isfinite(video).all(), 'Nonfinite model output'
        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        assert video.shape == (3, n + padding, height, width), video.shape
        print(f"CASE {case['case']}: inference {inferred-prepared:.3f}s; encoding", flush=True)
        images = official.tensor2video(video)
        official.save_video(images, str(intermediate), fps=float(fps), quality=5)
        encoded = time.perf_counter()
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(intermediate), "-i", str(src),
            "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "-1",
            "-vf", f"trim=end_frame={n},setpts=PTS-STARTPTS,setsar=1", "-r", str(fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart", str(dst),
        ], check=True)
        finished = time.perf_counter()
        del images, video, lq
        gc.collect()
        integrity = validate(src, dst, n, fps)
        breakdown = {
            "low_resolution_staging": staged - start,
            "input_tensor_preparation": prepared - staged,
            "inference": inferred - prepared,
            "materialization_and_encoding": encoded - inferred,
            "finalization": finished - encoded,
        }
        row = {
            "case": case["case"], "label": case["label"], "frames": n,
            "fps": str(fps), "duration_seconds": n / float(fps),
            "input_sha256": sha256(src), "output_sha256": sha256(dst),
            "output": str(dst.relative_to(output_dir)),
            "tail_padding": padding, "inference_frames": frames,
            "inference_seconds": breakdown["inference"],
            "inference_fps": n / breakdown["inference"],
            "processing_seconds_after_model_load": finished - start,
            "processing_seconds_excluding_lr_staging": finished - staged,
            "peak_reserved_vram_gib": peak_reserved, "peak_allocated_vram_gib": peak_allocated,
            "timing_breakdown_seconds": breakdown, "integrity": integrity,
            "input_pipeline": input_stats,
            "reference": case["flashvsr_v11"],
            "attention_backend": backend.metadata(),
        }
        if args.input_mode == 'eager':
            row["inference_delta_percent"] = (row["inference_seconds"] / row["reference"]["inference_seconds"] - 1) * 100
        results["cases"].append(row)
        write_json(case_dir / "metrics.json", row)
        write_json(output_dir / "results.json", results)
        print("RESULT " + json.dumps(row), flush=True)
    keys = ["inference_seconds", "inference_fps", "processing_seconds_after_model_load",
            "processing_seconds_excluding_lr_staging"]
    results["aggregate"] = {"mean_" + key: statistics.mean(c[key] for c in results["cases"]) for key in keys}
    results["aggregate"]["maximum_peak_reserved_vram_gib"] = max(c["peak_reserved_vram_gib"] for c in results["cases"])
    results["aggregate"]["all_integrity_checks_passed"] = all(all(c["integrity"].values()) for c in results["cases"])
    results["status"] = "completed"
    write_json(output_dir / "results.json", results)
    print("COMPLETED " + json.dumps(results["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
