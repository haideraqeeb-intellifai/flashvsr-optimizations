"""FFN-only W8A8 / W4A4 adapters, with native low precision GEMMs."""
import torch
import triton
import triton.language as tl

@triton.jit
def _row_quant(X, Q, S, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row*K + c, c<K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0)/448., 1.e-12)
    tl.store(Q + row*K + c, x/scale, c<K)
    tl.store(S + row, scale)

@triton.jit
def _partial_amax(X, P, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    x = tl.load(X+i, i<N, other=0).to(tl.float32)
    tl.store(P+tl.program_id(0), tl.max(tl.abs(x), 0))


def row_quant(x):
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((x.shape[0],1), device=x.device, dtype=torch.float32)
    _row_quant[(x.shape[0],)](x,q,scale,x.shape[1],triton.next_power_of_2(x.shape[1]))
    return q,scale


def global_scale(x):
    p = torch.empty((triton.cdiv(x.numel(),4096),), device=x.device, dtype=torch.float32)
    _partial_amax[(p.numel(),)](x,p,x.numel(),4096)
    return (2688. / p.amax().clamp_min(1.e-12)).reshape(1)


class QuantLinear(torch.nn.Module):
    def __init__(self, linear, mode):
        super().__init__()
        self.mode = mode
        self.out_features = linear.out_features
        self.register_buffer('bias', linear.bias.detach().clone() if linear.bias is not None else None)
        w = linear.weight.detach().contiguous()
        if mode == 'fp8':
            q,s = row_quant(w)
        elif mode == 'nvfp4':
            import flashinfer
            g = global_scale(w)
            q,s = flashinfer.nvfp4_quantize(w,g)
            self.register_buffer('global_sf',g)
        else:
            raise ValueError(mode)
        self.register_buffer('qweight',q)
        self.register_buffer('scale',s)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1,shape[-1]).contiguous()
        if self.mode == 'fp8':
            q,s = row_quant(x)
            out = torch._scaled_mm(q,self.qweight.t(),scale_a=s,scale_b=self.scale.t(),
                                   out_dtype=torch.bfloat16, use_fast_accum=False)
        else:
            import flashinfer
            g = global_scale(x)
            q,s = flashinfer.nvfp4_quantize(x,g)
            out = flashinfer.mm_fp4(q,self.qweight.t(),s,self.scale.t(),
                                    alpha=1./(g*self.global_sf),out_dtype=torch.bfloat16,
                                    backend='cutlass')
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1],self.out_features)


def install_ffn(pipe, mode):
    count = 0
    before = sum(p.numel()*p.element_size() for b in pipe.dit.blocks for p in b.ffn.parameters())
    if mode != 'bf16':
        for block in pipe.dit.blocks:
            for idx in (0,2):
                block.ffn[idx] = QuantLinear(block.ffn[idx],mode)
                count += 1
    after = sum(t.numel()*t.element_size() for b in pipe.dit.blocks
                for t in list(b.ffn.parameters())+list(b.ffn.buffers()))
    return {'mode':mode,'quantized_linears':count,'ffn_bytes_before':before,'ffn_bytes_after':after,
            'fp8':'E4M3 per-row W8A8; torch._scaled_mm; FP32 accumulation; BF16 output',
            'nvfp4':'E2M1 W4A4; 16-element E4M3 scales + dynamic tensor FP32 scale; FlashInfer CUTLASS; BF16 output',
            'scope':'Only dit.blocks.*.ffn.{0,2}; GELU, bias, attention, conditioning, decoder remain BF16'}
