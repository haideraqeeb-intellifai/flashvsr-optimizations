"""Native cuDNN FP8 3D convolution with BF16 output; no emulation fallback."""
import os
os.environ.setdefault('CUDNN_FRONTEND_CUDART_LIB_NAME', 'libcudart.so.13')
import torch
import triton
from .conv2d_fp8 import _quantize
from .ffn import _partial_amax

class FP8Conv3d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        import cudnn
        assert conv.groups == 1 and conv.padding == (0,0,0)
        self.c = cudnn
        self.handle = cudnn.create_handle()
        self.graphs = {}
        self.workspace = None
        self.stride, self.dilation = conv.stride, conv.dilation
        w = conv.weight.detach().float().contiguous(memory_format=torch.channels_last_3d)
        scale = (w.abs().amax()/448).clamp_min(1e-12).reshape(1,1,1,1,1)
        self.register_buffer('weight_scale', scale)
        self.register_buffer('qweight', (w/scale).to(torch.float8_e4m3fn))
        self.register_buffer('bias', conv.bias.detach().reshape(1,-1,1,1,1).contiguous() if conv.bias is not None else None)
    def quantize(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        partial = torch.empty((triton.cdiv(x.numel(),4096),), device=x.device, dtype=torch.float32)
        _partial_amax[(partial.numel(),)](x, partial, x.numel(), 4096)
        scale = (partial.amax()/448).clamp_min(1e-12).reshape(1,1,1,1,1)
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        _quantize[(triton.cdiv(x.numel(),4096),)](x,q,scale,x.numel(),4096)
        return q, scale
    def forward(self, x):
        q,s = self.quantize(x)
        c = self.c
        c.set_stream(self.handle, torch.cuda.current_stream().cuda_stream)
        key = (tuple(q.shape), tuple(q.stride()))
        shape = (q.shape[0], self.qweight.shape[0], *[(q.shape[i+2]-self.dilation[i]*(self.qweight.shape[i+2]-1)-1)//self.stride[i]+1 for i in range(3)])
        out = torch.empty(shape, device=x.device, dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
        if key not in self.graphs:
            g = c.pygraph(io_data_type=c.data_type.BFLOAT16, intermediate_data_type=c.data_type.FLOAT,
                          compute_data_type=c.data_type.FLOAT, handle=self.handle)
            X,W,SX,SW = [g.tensor_like(t) for t in (q,self.qweight,s,self.weight_scale)]
            y = g.conv_fprop(image=X,weight=W,padding=[0,0,0],stride=list(self.stride),dilation=list(self.dilation))
            y = g.mul(g.mul(y,SX),SW)
            B = g.tensor_like(self.bias) if self.bias is not None else None
            if B is not None: y = g.add(y,B)
            y.set_output(True).set_data_type(c.data_type.BFLOAT16).set_dim(list(shape)).set_stride(list(out.stride()))
            g.build([c.heur_mode.A,c.heur_mode.FALLBACK])
            size = g.get_workspace_size()
            if self.workspace is None or self.workspace.numel() < size:
                self.workspace = torch.empty(size, device=x.device, dtype=torch.uint8)
            self.graphs[key] = (g,X,W,SX,SW,B,y)
        g,X,W,SX,SW,B,Y = self.graphs[key]
        bindings = {X:q,W:self.qweight,SX:s,SW:self.weight_scale,Y:out}
        if B is not None: bindings[B] = self.bias
        g.execute(bindings,self.workspace)
        return out
