"""LQ causal Conv3D experiments, leaving the pinned vendor source untouched."""
import collections
import contextlib
import types
import torch
import torch.nn.functional as F
from einops import rearrange

MODES = ('baseline', 'bf16_layout', 'fp8_conv2', 'combined')

class OperationTimer:
    def __init__(self):
        self.enabled = False
        self.events = collections.defaultdict(list)
        self.layouts = {}
    @contextlib.contextmanager
    def scope(self, name):
        if not self.enabled:
            yield
            return
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record()
        self.events[name].append((a, b))
    def observe(self, name, x):
        if self.enabled and name not in self.layouts:
            self.layouts[name] = dict(shape=list(x.shape), stride=list(x.stride()),
                channels_last_3d=x.is_contiguous(memory_format=torch.channels_last_3d) if x.ndim == 5 else None)
    def results(self):
        return dict(operations={k:dict(seconds=sum(a.elapsed_time(b) for a,b in v)/1000, calls=len(v))
                               for k,v in self.events.items()}, layouts=self.layouts)


def install_lq(pipe, mode='baseline', instrument=False):
    assert mode in MODES
    m = pipe.dit.LQ_proj_in
    assert type(m).__name__ == 'Causal_LQ4x_Proj', type(m).__name__
    layout = mode in ('bf16_layout', 'combined')
    timer = OperationTimer()
    metadata = dict(mode=mode, autotune=False, channels_last_3d=layout,
                    fp8_conv2=mode in ('fp8_conv2', 'combined'), instrumented=instrument,
                    scope='LQ conditioning only; existing attention, FFNs and decoder unchanged')
    if mode == 'baseline' and not instrument:
        return metadata, timer
    def unwrap(conv):
        if hasattr(conv, 'module'):
            assert conv.onload_dtype == conv.computation_dtype
            assert conv.onload_device == conv.computation_device
            return conv.module
        return conv
    if layout:
        m.conv1.to(memory_format=torch.channels_last_3d)
        m.conv2.to(memory_format=torch.channels_last_3d)
    fp8 = None
    if metadata['fp8_conv2']:
        from .conv3d_fp8 import FP8Conv3d
        fp8 = FP8Conv3d(unwrap(m.conv2))
        m.add_module('fp8_conv2_engine', fp8)
        # Original Conv3D 2 is no longer needed for inference.
        unwrap(m.conv2).weight = None
        unwrap(m.conv2).bias = None
    def keep(x, name):
        if layout:
            with timer.scope('layout_conversion'):
                x = x.contiguous(memory_format=torch.channels_last_3d)
        timer.observe(name, x)
        return x
    def convolution(conv, x, cache, number):
        conv = unwrap(conv)
        pad = list(conv._padding)
        if cache is not None and pad[4] > 0:
            with timer.scope('cache_concat'):
                x = torch.cat([cache.to(x.device), x], dim=2)
                pad[4] -= cache.shape[2]
            x = keep(x, f'conv{number}_concat')
        with timer.scope('padding'):
            x = F.pad(x, pad, mode='replicate')
        x = keep(x, f'conv{number}_padded')
        with timer.scope(f'conv{number}'):
            if number == 2 and fp8 is not None:
                x = fp8(x)
            else:
                # Autotuning this causal Conv3D on the first streaming step
                # costs more than it saves, and each request starts a new process.
                previous = torch.backends.cudnn.benchmark
                try:
                    if layout: torch.backends.cudnn.benchmark = False
                    x = F.conv3d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
                finally:
                    torch.backends.cudnn.benchmark = previous
        return keep(x, f'conv{number}_output')
    def stream(self, video_clip):
        with timer.scope('lq_total'):
            first = self.clip_idx == 0
            if first:
                with timer.scope('initial_frame_repeat'):
                    video_clip = torch.cat([video_clip[:,:,:1].repeat(1,1,3,1,1), video_clip], dim=2)
            with timer.scope('pixel_shuffle'):
                x = self.pixel_shuffle(video_clip)
            x = keep(x, 'pixel_shuffle_output')
            with timer.scope('cache_copy'):
                cache1 = x[:,:,-2:].clone(memory_format=torch.channels_last_3d if layout else torch.preserve_format)
            timer.observe('cache1', cache1)
            x = convolution(self.conv1, x, self.cache['conv1'], 1)
            self.cache['conv1'] = cache1
            with timer.scope('norm1'): x = self.norm1(x)
            x = keep(x, 'norm1_output')
            with timer.scope('silu1'): x = self.act1(x)
            x = keep(x, 'silu1_output')
            with timer.scope('cache_copy'):
                cache2 = x[:,:,-2:].clone(memory_format=torch.channels_last_3d if layout else torch.preserve_format)
            timer.observe('cache2', cache2)
            if first:
                self.cache['conv2'] = cache2
                self.clip_idx += 1
                return None
            x = convolution(self.conv2, x, self.cache['conv2'], 2)
            self.cache['conv2'] = cache2
            with timer.scope('norm2'): x = self.norm2(x)
            x = keep(x, 'norm2_output')
            with timer.scope('silu2'): x = self.act2(x)
            x = keep(x, 'silu2_output')
            with timer.scope('projection_reshape'):
                x = rearrange(x, 'b c f h w -> b (f h w) c')
            with timer.scope('projection'):
                outputs = [layer(x) for layer in self.linear_layers]
            self.clip_idx += 1
            return outputs
    m.stream_forward = types.MethodType(stream, m)
    return metadata, timer
