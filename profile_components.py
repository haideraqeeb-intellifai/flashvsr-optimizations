"""Measure Tiny-Long components with CUDA event boundaries.

Uses an already staged baseline clip; does not encode or overwrite videos.
Nested timings are inclusive and must not be added to their parent timings.
"""
import argparse
import collections
import contextlib
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path[:0] = [str(ROOT / "vendor/FlashVSR"), str(ROOT / "vendor/FlashVSR/examples/WanVSR")]

import torch
from diffsynth.pipelines import flashvsr_tiny_long as pipeline_module
from diffsynth.models import wan_video_dit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="female_default")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--attention", choices=("baseline", "jev_replay", "headwise_bf16"), default="baseline")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--ffn", choices=('bf16','fp8','nvfp4'), default='bf16')
    from decoder_optimizations import MODES
    parser.add_argument('--decoder', choices=MODES, default='baseline')
    from lq_optimizations import MODES as LQ_MODES
    parser.add_argument('--lq', choices=LQ_MODES, default='baseline')
    parser.add_argument('--decoder-layers', action='store_true')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location(
        "official_profile", ROOT / "vendor/FlashVSR/examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from attention_backends import AttentionBackend
    if args.attention == "headwise_bf16":
        from sparse_study_backends import SparseStudyBackend
        backend = SparseStudyBackend(args.attention)
    else:
        backend = AttentionBackend("jev_sparse" if args.attention == "jev_replay" else "baseline")
    saved_decision = None
    if args.attention == "jev_replay":
        saved_path = ROOT / "outputs/attention/jev_sparse" / args.case / "metrics.json"
        saved_metrics = json.loads(saved_path.read_text())
        assert saved_metrics["case"] == args.case
        saved_decision = saved_metrics["attention_backend"]["jev"]
        rates = saved_decision["applied_keep_percent"]
        assert len(rates) == 30 and all(r in (1, 3, 5, 10) for r in rates)
    os.chdir(ROOT / "outputs/lq-study/baseline/runtime/examples/WanVSR")
    with torch.no_grad():
        pipe = official.init_pipeline()
        from ffn_quantization import install_ffn
        ffn_metadata = install_ffn(pipe,args.ffn)
        from decoder_optimizations import install_decoder_optimization
        decoder_metadata = install_decoder_optimization(pipe, args.decoder)
        if args.decoder in ('fp8', 'nvfp4'):
            ffn_metadata['scope'] = 'Only dit.blocks.*.ffn.{0,2}; decoder precision is configured separately and recorded in decoder metadata'
    from lq_optimizations import install_lq
    lq_metadata, lq_timer = install_lq(pipe, args.lq, instrument=True)
    source = ROOT / "outputs/lq-study/baseline" / args.case / "low_resolution_padded.mp4"
    with open(output.with_suffix(".prepare.log"), "w") as log, contextlib.redirect_stdout(log):
        lq, height, width, frames, _ = official.prepare_input_tensor(
            str(source), scale=4, dtype=torch.bfloat16, device="cuda")
    if args.warmup:
        # Match the study: warm up with the original mask before setting JEV rates.
        with torch.no_grad():
            warm_video = pipe(prompt="", negative_prompt="", cfg_scale=1.0,
                              num_inference_steps=1, seed=2026,
                              LQ_video=lq[:, :, :49].contiguous(), num_frames=49,
                              height=height, width=width, is_full_block=False,
                              if_buffer=True, topk_ratio=2.0 * 768 * 1280 / (height * width),
                              kv_ratio=3.0, local_range=11, color_fix=False)
        torch.cuda.synchronize()
        del warm_video
    backend.reset()
    if saved_decision is not None:
        backend.rates = list(rates)
    mask_samples = {}
    original_mask = wan_video_dit.generate_draft_block_mask

    def sample_mask(*positional, **keywords):
        mask = original_mask(*positional, **keywords)
        shape = str(tuple(mask.shape))
        if shape not in mask_samples:
            # Retain only the first mask for each shape; count after inference.
            mask_samples[shape] = mask
        return mask

    wan_video_dit.generate_draft_block_mask = sample_mask
    events = collections.defaultdict(list)
    input_dtypes = {}
    decoder_layers = {}

    def timed(owner, attribute, name, aggregate=None):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def wrapper(*positional, **keywords):
            if name not in input_dtypes:
                tensors = [v for v in (*positional, *keywords.values()) if torch.is_tensor(v)]
                input_dtypes[name] = sorted({str(v.dtype) for v in tensors})
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*positional, **keywords)
            end.record()
            events[name].append((start, end))
            if aggregate is not None:
                events[aggregate].append((start, end))
            return result

        setattr(owner, attribute, wrapper)

    timed(pipe.dit.LQ_proj_in, "stream_forward", "lq_conditioning")
    timed(pipeline_module, "model_fn_wan_video", "dit_total")
    timed(pipe.TCDecoder, "decode_video", "decoder_total")
    if args.detail:
        for block in pipe.dit.blocks:
            timed(block.ffn, "forward", "dit_ffn")
            timed(block.self_attn, "forward", "dit_self_attention")
            timed(block.cross_attn, "forward", "dit_cross_attention")
            timed(block.self_attn.attn, "forward", "dit_sparse_attention_kernel_path")
            for key in ["q", "k", "v", "o"]:
                timed(getattr(block.self_attn, key), "forward", "dit_self_attention_projections")
        timed(wan_video_dit, "rope_apply", "dit_rope")
        timed(wan_video_dit, "generate_draft_block_mask", "dit_mask_selection")
        for layer_name, module in pipe.TCDecoder.named_modules():
            if isinstance(module, torch.nn.Conv2d) or getattr(module, "is_fp8_conv", False) or getattr(module, "is_nvfp4_conv", False):
                if args.decoder_layers:
                    key = 'decoder_layer.' + layer_name
                    def observe(mod, inputs, name=layer_name):
                        if name not in decoder_layers:
                            x = inputs[0]
                            decoder_layers[name] = dict(input_shape=list(x.shape),
                                input_stride=list(x.stride()),
                                channels_last=x.is_contiguous(memory_format=torch.channels_last),
                                implementation="CUTLASS NVFP4 adapter" if getattr(mod, "is_nvfp4_conv", False) else "cuDNN FP8 adapter" if getattr(mod, "is_fp8_conv", False) else "PyTorch BF16 Conv2d",
                                kernel_size=list(mod.kernel_size),
                                in_channels=mod.in_channels, out_channels=mod.out_channels)
                    module.register_forward_pre_hook(observe)
                    timed(module, 'forward', key, 'decoder_convolutions')
                else:
                    timed(module, "forward", "decoder_convolutions")
    weights = {}
    for name, model in [("dit_including_lq", pipe.dit), ("lq", pipe.dit.LQ_proj_in), ("decoder", pipe.TCDecoder)]:
        counts = collections.Counter()
        for parameter in model.parameters():
            counts[str(parameter.dtype)] += parameter.numel()
        weights[name] = dict(counts)
    torch.cuda.synchronize()
    lq_timer.enabled = True
    begin = time.perf_counter()
    with torch.no_grad():
        video = pipe(prompt="", negative_prompt="", cfg_scale=1.0,
                     num_inference_steps=1, seed=2026, LQ_video=lq,
                     num_frames=frames, height=height, width=width,
                     is_full_block=False, if_buffer=True,
                     topk_ratio=2.0 * 768 * 1280 / (height * width),
                     kv_ratio=3.0, local_range=11, color_fix=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    assert tuple(video.shape) == (3, frames - 4, height, width)
    timings = {name: {"seconds": sum(s.elapsed_time(e) for s, e in pairs) / 1000,
                      "calls": len(pairs)} for name, pairs in events.items()}
    result = {
        "case": args.case, "detail": args.detail,
        "attention": args.attention, "warmup_49_frames": args.warmup,
        "ffn": ffn_metadata,
        "decoder": decoder_metadata,
        "lq": lq_metadata, "lq_operations": lq_timer.results(),
        "decoder_layers": decoder_layers,
        "replayed_keep_percent": backend.rates,
        "historical_decision_wait_seconds_excluded": saved_decision["api_wall_seconds"] if saved_decision else None,
        "attention_calls": dict(backend.calls),
        "attention_backend": backend.metadata(),
        "first_mask_per_shape": {shape: {"selected": int(mask.sum().item()),
                                          "total": mask.numel(),
                                          "fraction": float(mask.float().mean().item())}
                                 for shape, mask in mask_samples.items()},
        "source": str(source), "seed": 2026,
        "input_shape": list(lq.shape), "output_shape": list(video.shape),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "inference_wall_seconds": elapsed, "cuda_event_timings": timings,
        "weight_dtype_parameter_counts": weights, "input_dtypes": input_dtypes,
        "notes": ["Diagnostic rerun, not the original benchmark timing.",
                  "CUDA event intervals include device idle gaps between launches; not kernel-only FLOP counts.",
                  "Nested groups overlap their parents. Instrumentation adds overhead.",
                  "Model startup, preprocessing, and video encoding excluded. CPU output transfer included in wall time."]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
