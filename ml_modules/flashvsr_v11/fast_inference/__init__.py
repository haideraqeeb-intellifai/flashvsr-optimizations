"""Install the measured FlashVSR v1.1 fast-inference configuration."""

import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")

from .attention import install_headwise_attention
from .decoder import install_decoder_optimization
from .ffn import install_ffn
from .lq import install_lq


def install_fast_inference(pipe):
    attention = install_headwise_attention(pipe)
    ffn = install_ffn(pipe, "nvfp4")
    if ffn["quantized_linears"] != 60:
        raise RuntimeError(f"Expected 60 NVFP4 FFN linears, found {ffn['quantized_linears']}")
    decoder = install_decoder_optimization(pipe, "fp8")
    if decoder["quantized_convolutions"] != 35:
        raise RuntimeError(f"Expected 35 FP8 decoder convolutions, found {decoder['quantized_convolutions']}")
    lq, _ = install_lq(pipe, "combined")
    if not lq["fp8_conv2"] or not lq["channels_last_3d"] or lq["autotune"]:
        raise RuntimeError("FlashVSR LQ conditioning did not install its FP8/layout configuration")
    return {"attention": attention, "ffn": ffn, "decoder": decoder, "lq_conditioning": lq}


def check_fast_dependencies(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("Fast FlashVSR requires CUDA")
    if torch.cuda.get_device_capability() < (12, 0):
        raise RuntimeError("Fast FlashVSR requires the RTX 5090 / SM120 kernels")
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        from torch.utils.cpp_extension import CUDA_HOME
        cuda_home = CUDA_HOME
    if cuda_home and not (Path(cuda_home) / "include/cublasLt.h").is_file() and Path("/usr/local/cuda/include/cublasLt.h").is_file():
        cuda_home = "/usr/local/cuda"
    if not cuda_home or not (Path(cuda_home) / "include/cublasLt.h").is_file():
        raise RuntimeError("Fast FlashVSR needs CUDA toolkit headers (including cublasLt.h) for NVFP4 JIT compilation")
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    try:
        import cudnn
        import flashinfer
        import triton
    except ImportError as exc:
        raise RuntimeError("Fast FlashVSR needs FlashInfer, cuDNN frontend, and Triton in its runtime") from exc
    if not shutil.which("ninja", path=os.pathsep.join((os.path.dirname(sys.executable), os.environ.get("PATH", "")))):
        raise RuntimeError("Fast FlashVSR needs Ninja on PATH to compile FlashInfer's native NVFP4 kernels")
    if not hasattr(flashinfer, "nvfp4_quantize") or not hasattr(flashinfer, "mm_fp4"):
        raise RuntimeError("Installed FlashInfer does not provide native NVFP4 quantization and GEMM")
    if not hasattr(cudnn, "pygraph"):
        raise RuntimeError("Installed cuDNN frontend does not provide native convolution graphs")
    return {"torch": torch.__version__, "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "cudnn_frontend": getattr(cudnn, "__version__", "unknown"), "triton": triton.__version__,
            "compute_capability": list(torch.cuda.get_device_capability())}
