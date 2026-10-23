"""Pinned FlashVSR assets and worker entry point; safe to import without Torch/CUDA."""

import os
from pathlib import Path

SOURCE_URL = "https://github.com/OpenImagingLab/FlashVSR.git"
SOURCE_REVISION = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
MODEL_ID = "JunhaoZhuang/FlashVSR-v1.1"
MODEL_REVISION = "27561b186ded3402d7c975f4fd722e2885b6135f"
MODEL_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def cache_root():
    return Path(os.environ.get("FLASHVSR_V11_CACHE", REPO_ROOT / "cache/flashvsr_v11")).resolve()


def run_official(arguments, log_path):
    """Reuse FlashVSR and its CPU model cache in the LTX worker."""
    # ImageIO's FFmpeg wrapper otherwise forks to install signal handlers
    # after the parent has initialized CUDA on a previous SR request.
    os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
    from .inprocess import run_official as run_in_process
    return run_in_process(arguments, log_path)
