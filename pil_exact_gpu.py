"""GPU replica of Pillow 11.2.1 RGB BICUBIC for the 426x240 -> 1704x960 resize.

Pillow's 8-bit path uses two separable passes. Each pass rounds its normalized
double-precision weights to signed 22-bit fixed point, accumulates integer
products with a half-unit bias, shifts, and clips to uint8. These kernels use
the same coefficients and integer operations so the intermediate horizontal
image is rounded before the vertical pass.
"""

from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl


PRECISION_BITS = 22


def _cubic(x):
    x = abs(x)
    if x < 1.0:
        return ((1.5 * x - 2.5) * x * x + 1.0)
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    return 0.0


def _coefficients(in_size, out_size):
    scale = in_size / out_size
    support = 2.0
    starts = np.empty(out_size, dtype=np.int32)
    weights = np.zeros((out_size, 4), dtype=np.int32)
    for pos in range(out_size):
        center = (pos + 0.5) * scale
        xmin = max(0, int(center - support + 0.5))
        xmax = min(in_size, int(center + support + 0.5))
        taps = [_cubic(x + xmin - center + 0.5) for x in range(xmax - xmin)]
        total = sum(taps)
        starts[pos] = xmin
        for j, value in enumerate(taps):
            normalized = value / total if total != 0.0 else value
            scaled = normalized * (1 << PRECISION_BITS)
            weights[pos, j] = int(scaled - 0.5 if scaled < 0 else scaled + 0.5)
    return starts, weights


@lru_cache(maxsize=4)
def _device_coefficients(device):
    hx, hw = _coefficients(426, 1704)
    vy, vw = _coefficients(240, 960)
    return tuple(torch.from_numpy(value).to(device) for value in (hx, hw, vy, vw))


@triton.jit
def _horizontal(src, dst, starts, weights, total: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < total
    channel = index % 3
    x = index // 3 % 1704
    y = index // (3 * 1704) % 240
    batch = index // (3 * 1704 * 240)
    first = tl.load(starts + x, valid, 0)
    acc = tl.full((block,), 1 << 21, tl.int32)
    for j in tl.static_range(4):
        coeff = tl.load(weights + x * 4 + j, valid, 0)
        source_x = first + j
        pixel = tl.load(src + ((batch * 240 + y) * 426 + source_x) * 3 + channel,
                        valid & (source_x < 426), 0).to(tl.int32)
        acc += pixel * coeff
    result = tl.minimum(tl.maximum(acc >> 22, 0), 255).to(tl.uint8)
    tl.store(dst + index, result, valid)


@triton.jit
def _vertical(src, dst, starts, weights, total: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < total
    channel = index % 3
    x = index // 3 % 1704
    y = index // (3 * 1704) % 960
    batch = index // (3 * 1704 * 960)
    first = tl.load(starts + y, valid, 0)
    acc = tl.full((block,), 1 << 21, tl.int32)
    for j in tl.static_range(4):
        coeff = tl.load(weights + y * 4 + j, valid, 0)
        source_y = first + j
        pixel = tl.load(src + ((batch * 240 + source_y) * 1704 + x) * 3 + channel,
                        valid & (source_y < 240), 0).to(tl.int32)
        acc += pixel * coeff
    result = tl.minimum(tl.maximum(acc >> 22, 0), 255).to(tl.uint8)
    tl.store(dst + index, result, valid)


def resize_exact_gpu(frames):
    """Return cropped RGB uint8 frames [T,896,1664,3] on CUDA."""
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (240, 426, 3) or frames.dtype != torch.uint8:
        raise ValueError("Expected [T,240,426,3] uint8 input")
    frames = frames.to("cuda").contiguous()
    batch = frames.shape[0]
    hx, hw, vy, vw = _device_coefficients(frames.device)
    temp = torch.empty((batch, 240, 1704, 3), device=frames.device, dtype=torch.uint8)
    out = torch.empty((batch, 960, 1704, 3), device=frames.device, dtype=torch.uint8)
    block = 256
    horizontal_count = temp.numel()
    vertical_count = out.numel()
    _horizontal[(triton.cdiv(horizontal_count, block),)](
        frames, temp, hx, hw, horizontal_count, block)
    _vertical[(triton.cdiv(vertical_count, block),)](
        temp, out, vy, vw, vertical_count, block)
    return out[:, 32:928, 20:1684, :]


@lru_cache(maxsize=4)
def _normalization_lut(device):
    pixels = torch.arange(256, dtype=torch.float32)
    # Evaluate on CPU in the same order as upstream pil_to_tensor_neg1_1.
    return (pixels / 255.0 * 2.0 - 1.0).to(torch.bfloat16).to(device)


@triton.jit
def _normalize(src, dst, lut, stride_b: tl.constexpr, stride_y: tl.constexpr,
               stride_x: tl.constexpr, total: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < total
    x = index % 1664
    y = index // 1664 % 896
    channel = index // (1664 * 896) % 3
    batch = index // (1664 * 896 * 3)
    pixel = tl.load(src + batch * stride_b + y * stride_y + x * stride_x + channel,
                    valid, 0).to(tl.int32)
    value = tl.load(lut + pixel, valid, 0)
    tl.store(dst + index, value, valid)


def normalize_exact_gpu(cropped):
    """Return [T,3,896,1664] BF16 using upstream's exact CPU pixel mapping."""
    if cropped.ndim != 4 or tuple(cropped.shape[1:]) != (896, 1664, 3) or cropped.dtype != torch.uint8:
        raise ValueError("Expected [T,896,1664,3] uint8 input")
    if cropped.device.type != "cuda":
        raise ValueError("Input must be on CUDA")
    output = torch.empty((cropped.shape[0], 3, 896, 1664),
                         dtype=torch.bfloat16, device=cropped.device)
    total = output.numel()
    block = 256
    _normalize[(triton.cdiv(total, block),)](
        cropped, output, _normalization_lut(cropped.device),
        cropped.stride(0), cropped.stride(1), cropped.stride(2), total, block)
    return output
