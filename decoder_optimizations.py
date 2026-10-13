"""Decoder-only execution options; trained weights and streaming logic are preserved."""
import functools
import time

import torch

MODES = ('baseline', 'autotune', 'channels_last', 'autotune_channels_last', 'fp8', 'nvfp4')


def install_decoder_optimization(pipe, mode):
    if mode not in MODES:
        raise ValueError(mode)
    started = time.perf_counter()
    decoder = pipe.TCDecoder
    autotune = mode in ('autotune', 'autotune_channels_last', 'fp8', 'nvfp4')
    channels_last = mode in ('channels_last', 'autotune_channels_last', 'fp8', 'nvfp4')
    convolutions = [(name, m) for name, m in decoder.named_modules()
                    if isinstance(m, torch.nn.Conv2d)]
    if channels_last:
        decoder.to(memory_format=torch.channels_last)

        def ensure_layout(module, inputs):
            # TGrow reshapes and streaming slices can break layout propagation.
            # contiguous is a no-op when the requested format is already present.
            return (inputs[0].contiguous(memory_format=torch.channels_last), *inputs[1:])

        for _, module in convolutions:
            module.register_forward_pre_hook(ensure_layout)

    if autotune:
        original = decoder.decode_video

        @functools.wraps(original)
        def decode(*args, **kwargs):
            previous = torch.backends.cudnn.benchmark
            try:
                torch.backends.cudnn.benchmark = True
                return original(*args, **kwargs)
            finally:
                torch.backends.cudnn.benchmark = previous

        decoder.decode_video = decode

    fp8_metadata = {}
    if mode == 'fp8':
        from decoder_fp8 import install_fp8
        fp8_metadata = install_fp8(decoder)
        for module in decoder.modules():
            if getattr(module, "is_fp8_conv", False):
                module.register_forward_pre_hook(ensure_layout)

    if mode == 'nvfp4':
        from decoder_nvfp4 import install_nvfp4
        fp8_metadata = install_nvfp4(decoder)
        for module in decoder.modules():
            if getattr(module, 'is_nvfp4_conv', False):
                module.register_forward_pre_hook(ensure_layout)

    return {**fp8_metadata, 'mode': mode, 'scope': f'TCDecoder only; internal {mode.upper()} convolutions' if mode in ('fp8', 'nvfp4') else 'TCDecoder only; BF16 weights/activations',
            'autotune': autotune, 'channels_last': channels_last,
            'conv2d_modules': len(convolutions),
            'layout_policy': 'Conv2d weights and inputs channels_last; conversion costs included' if channels_last else 'unchanged',
            'setup_seconds': time.perf_counter() - started}
