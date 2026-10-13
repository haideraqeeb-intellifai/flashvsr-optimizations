"""Native cuDNN E4M3 W8A8 decoder convolutions with dynamic activation scales."""
import os
os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")
import torch
import triton
import triton.language as tl
from ffn_quantization import _partial_amax

@triton.jit
def _quantize(X,Q,S,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0).to(tl.float32)
    scale=tl.load(S)
    tl.store(Q+i,tl.minimum(tl.maximum(x/scale,-448.),448.),i<N)


def quantize_activation(x):
    x=x.contiguous(memory_format=torch.channels_last)
    partial=torch.empty((triton.cdiv(x.numel(),4096),),device=x.device,dtype=torch.float32)
    _partial_amax[(partial.numel(),)](x,partial,x.numel(),4096)
    scale=(partial.amax()/448.).clamp_min(1e-12).reshape(1,1,1,1)
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    _quantize[(triton.cdiv(x.numel(),4096),)](x,q,scale,x.numel(),4096)
    return q,scale


class ConvEngineCache:
    def __init__(self):
        import cudnn
        self.cudnn=cudnn
        self.handle=cudnn.create_handle()
        self.graphs={}
        self.workspace=None

    def get(self,x,w,sx,sw,bias,padding):
        c=self.cudnn
        c.set_stream(self.handle,torch.cuda.current_stream().cuda_stream)
        key=(tuple(x.shape),tuple(x.stride()),tuple(w.shape),padding,bias is not None)
        if key not in self.graphs:
            graph=c.pygraph(io_data_type=c.data_type.BFLOAT16,
                intermediate_data_type=c.data_type.FLOAT,compute_data_type=c.data_type.FLOAT,handle=self.handle)
            X=graph.tensor_like(x);W=graph.tensor_like(w)
            SX=graph.tensor_like(sx);SW=graph.tensor_like(sw)
            y=graph.conv_fprop(image=X,weight=W,padding=[padding]*2,stride=[1,1],dilation=[1,1])
            y=graph.mul(y,SX);y=graph.mul(y,SW)
            B=graph.tensor_like(bias) if bias is not None else None
            if B is not None:y=graph.add(y,B)
            shape=(x.shape[0],w.shape[0],x.shape[2],x.shape[3])
            stride=(shape[1]*shape[2]*shape[3],1,shape[3]*shape[1],shape[1])
            y.set_output(True).set_data_type(c.data_type.BFLOAT16).set_dim(shape).set_stride(stride)
            graph.build([c.heur_mode.A,c.heur_mode.FALLBACK])
            size=graph.get_workspace_size()
            if self.workspace is None or self.workspace.numel()<size:
                self.workspace=torch.empty(size,device=x.device,dtype=torch.uint8)
            self.graphs[key]=(graph,X,W,SX,SW,B,y,shape)
        return self.graphs[key]


class FP8Conv2d(torch.nn.Module):
    is_fp8_conv=True
    def __init__(self,conv,cache):
        super().__init__()
        assert conv.groups==1 and conv.stride==(1,1) and conv.dilation==(1,1)
        assert conv.padding_mode=='zeros'
        self.in_channels=conv.in_channels;self.out_channels=conv.out_channels
        self.kernel_size=conv.kernel_size;self.padding=conv.padding
        self.cache=cache
        w=conv.weight.detach().float().contiguous(memory_format=torch.channels_last)
        scale=(w.abs().amax()/448.).clamp_min(1e-12).reshape(1,1,1,1)
        self.register_buffer('qweight',(w/scale).to(torch.float8_e4m3fn))
        self.register_buffer('weight_scale',scale)
        self.register_buffer('bias',conv.bias.detach().reshape(1,-1,1,1).contiguous() if conv.bias is not None else None)

    def forward(self,x):
        q,s=quantize_activation(x)
        graph,X,W,SX,SW,B,Y,shape=self.cache.get(q,self.qweight,s,self.weight_scale,self.bias,self.padding[0])
        out=torch.empty(shape,device=x.device,dtype=torch.bfloat16,memory_format=torch.channels_last)
        bindings={X:q,W:self.qweight,SX:s,SW:self.weight_scale,Y:out}
        if B is not None:bindings[B]=self.bias
        graph.execute(bindings,self.cache.workspace)
        return out


def install_fp8(decoder):
    cache=ConvEngineCache()
    replacements=[];retained=[]
    # Keep the input and RGB output convolutions BF16 for a conservative first trial.
    layers=[(name,m) for name,m in decoder.named_modules() if isinstance(m,torch.nn.Conv2d)]
    before=sum(p.numel()*p.element_size() for p in decoder.parameters())
    for index,(name,module) in enumerate(layers):
        if index in (0,len(layers)-1):
            retained.append(name);continue
        parent_name,_,child=name.rpartition('.')
        parent=decoder.get_submodule(parent_name) if parent_name else decoder
        setattr(parent,child,FP8Conv2d(module,cache))
        replacements.append(name)
    after=sum(t.numel()*t.element_size() for t in list(decoder.parameters())+list(decoder.buffers()))
    return dict(quantized_convolutions=len(replacements),quantized_layers=replacements,
        retained_bf16_layers=retained,weight_bytes_before=before,weight_bytes_after=after,
        precision='E4M3 W8A8; per-tensor static weight scales and dynamic activation scales; FP32 accumulation/descaling; BF16 output and bias',
        backend='cuDNN frontend native FP8 convolution, heuristic A/fallback plan selection',
        calibration='No calibration dataset; activation max is recomputed per convolution call')
