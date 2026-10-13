"""Sparse-attention experiments, optionally replaying saved JEV decisions locally."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from attention_backends import AttentionBackend, ROOT

NAMES = ('sparge2_10', 'headwise_bf16', 'headwise_sparge2', 'triton_int8_fp8_10',
         'jev_replay_bf16', 'jev_replay_sparge2')
SAGE2_NAMES = ('sparge2_10', 'headwise_sparge2', 'jev_replay_sparge2')
BF16_NAMES = ('headwise_bf16', 'jev_replay_bf16')
JEV_NAMES = ('jev_replay_bf16', 'jev_replay_sparge2')
POLICY = ROOT / 'config/head-policy.json'


def headwise_mask(batch_size, nheads, seqlen, q_w, k_w, local_attn_mask, rates):
    """Same pooled scores/local eligibility/strict threshold as upstream; per-head budgets."""
    assert batch_size == 1 and len(rates) == nheads
    q = q_w.mean(1).reshape(-1, nheads, q_w.shape[-1] // nheads).permute(1, 0, 2)
    k = k_w.mean(1).reshape(-1, nheads, k_w.shape[-1] // nheads).permute(1, 0, 2)
    scores = torch.einsum('hld,hmd->hlm', q, k) / math.sqrt(q.shape[-1])
    eligible = local_attn_mask.repeat(scores.shape[1] // local_attn_mask.shape[0],
                                     scores.shape[2] // local_attn_mask.shape[1])
    # Upstream adds an FP32 local mask before softmax, promoting pooled BF16 scores.
    scores = scores.float().masked_fill(~eligible, -float('inf'))
    flat = scores.softmax(-1).reshape(nheads * seqlen, -1)
    counts = [min(flat.shape[1]-1, max(1, math.ceil(flat.shape[1]*r/100)))
              for r in rates for _ in range(seqlen)]
    ranked = torch.topk(flat, max(counts)+1, dim=-1).values
    indices = torch.tensor(counts, device=q.device, dtype=torch.long)[:, None]
    threshold = ranked.gather(1, indices)
    return (flat > threshold).reshape(1, nheads, scores.shape[1], scores.shape[2])


class FlashInferSparse:
    def __init__(self):
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
        from sageattention.quant import per_warp_int8
        self.kernel = bsa_attn_sm120_blk64_sage_fwd
        self.quant_qk = per_warp_int8
        self.workspace = torch.empty(4096, device='cuda', dtype=torch.uint8)

    def __call__(self, q, k, v, mask):
        # Q/K quantization does not absorb softmax scale in this API.
        qi, qs, ki, ks = self.quant_qk(q, k, km=k.mean(-2, keepdim=True),
                                     BLKQ=128, WARPQ=32, BLKK=64, tensor_layout='HND')
        # Unpermuted HDS V is required by FlashInfer (different from Sage CUDA's layout).
        vs = v.float().abs().amax(-2).clamp_min(1e-8) / 448.0
        vf = (v.float() / vs.unsqueeze(-2)).clamp(-448, 448).to(torch.float8_e4m3fn)
        vf = vf.transpose(-1, -2).contiguous()
        m64 = mask.repeat_interleave(2, -2).repeat_interleave(2, -1)
        counts = m64.sum(-1, dtype=torch.int32).contiguous()
        # Keep a fixed-capacity LUT to avoid a device-to-host max-count sync.
        columns = torch.arange(m64.shape[-1], device=q.device, dtype=torch.int32)
        lut = torch.where(m64, columns, m64.shape[-1]).sort(-1).values.contiguous()
        out = torch.empty_like(q)
        return self.kernel(qi, ki, vf, qs, ks, vs.contiguous(), lut, 0,
                           q2k_block_nums=counts, out=out,
                           tma_descriptor_workspace=self.workspace)


class SparseStudyBackend(AttentionBackend):
    def __init__(self, name):
        assert name in NAMES
        super().__init__('baseline')
        self.name = name
        self.policy = json.loads(POLICY.read_text()) if name in ('headwise_bf16', 'headwise_sparge2') else None
        self.head_rates = self.policy['rates_percent'] if self.policy else [[10]*12 for _ in range(30)]
        self.mask_stats = None
        if name in SAGE2_NAMES:
            sys.path.insert(0, str(ROOT / 'vendor/SpargeAttn'))
            from spas_sage_attn import block_sparse_sage2_attn_cuda
            from spas_sage_attn.core import SAGE2PP_ENABLED
            assert SAGE2PP_ENABLED, 'Sparse Sage2++ unavailable; no fallback allowed'
            self.kernel = block_sparse_sage2_attn_cuda
        elif name == 'triton_int8_fp8_10':
            from sparse_int8_fp8 import TritonSparseFP8
            self.kernel = TritonSparseFP8()
        from diffsynth.models import wan_video_dit as dit
        old_mask = dit.generate_draft_block_mask
        def select(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None):
            layer = self.calls['self'] % 30
            if self.policy is not None:
                return headwise_mask(batch_size, nheads, seqlen, q_w, k_w,
                                     local_attn_mask, self.head_rates[layer])
            pairs = (q_w.shape[0] // seqlen) * k_w.shape[0]
            if self.name in JEV_NAMES:
                assert self.decisions is not None, 'Load saved JEV policy before inference'
                rate = self.rates[layer] / 100
            else:
                rate = 0.1
            return old_mask(batch_size, nheads, seqlen, q_w, k_w,
                            max(1, math.ceil(pairs * rate)), local_attn_mask)
        dit.generate_draft_block_mask = select

    def before_inference(self, case):
        if self.name not in JEV_NAMES:
            return
        path = ROOT/'outputs/attention/jev_sparse'/case['case']/'metrics.json'
        raw = path.read_bytes()
        saved = json.loads(raw)
        assert saved['case'] == case['case']
        assert saved['input_sha256'] == case['input']['sha256']
        self.rates = list(saved['attention_backend']['jev']['applied_keep_percent'])
        assert len(self.rates) == 30 and all(r in (1,3,5,10) for r in self.rates)
        self.head_rates = [[rate]*12 for rate in self.rates]
        self.decisions = {'applied_keep_percent':self.rates, 'api_wall_seconds':0,
                          'replay_source':str(path), 'replay_source_sha256':hashlib.sha256(raw).hexdigest()}

    def reset(self):
        super().reset()
        self.mask_stats = None

    def metadata(self):
        return {'backend': self.name, 'calls': dict(self.calls), 'shapes': dict(self.shapes),
                'cross_attention': 'unchanged torch SDPA', 'remote_calls': 0,
                'rates_percent': self.head_rates, 'policy': self.policy,
                'jev': self.decisions,
                'mask_stats_selected_total': self.mask_stats.tolist() if self.mask_stats is not None else None,
                'precision': 'BF16' if self.name in BF16_NAMES else 'INT8 QK / FP8 PV / BF16 output',
                'mask': 'Original pooled score and local eligibility; changed retention budgets',
                'pv_threshold': 'disabled (1e6)' if self.name in SAGE2_NAMES else None}

    def forward(self, q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
        self.calls['self' if attention_mask is not None else 'cross'] += 1
        if attention_mask is None:
            return self.original(q,k,v,num_heads,compatibility_mode,attention_mask,return_KV)
        self.shapes[str((tuple(q.shape), tuple(k.shape), num_heads))] += 1
        # GPU accumulation only; host read occurs outside inference in metadata().
        selected = attention_mask.sum()
        stat = torch.stack((selected, selected.new_tensor(attention_mask.numel())))
        self.mask_stats = stat if self.mask_stats is None else self.mask_stats + stat
        if self.name in BF16_NAMES:
            return self.original(q,k,v,num_heads,compatibility_mode,attention_mask,return_KV)
        b,n,c = q.shape
        q,k,v = [x.reshape(b,-1,num_heads,c//num_heads).transpose(1,2).contiguous() for x in (q,k,v)]
        if self.name in SAGE2_NAMES:
            mask = attention_mask.repeat_interleave(2, -1).to(torch.int8).contiguous()
            out = self.kernel(q,k,v,mask_id=mask,tensor_layout='HND',pvthreshd=1e6)
            # Upstream Sparse Sage2 leaves empty query tiles uninitialized.
            valid = attention_mask.any(-1).repeat_interleave(128, -1).unsqueeze(-1)
            out.masked_fill_(~valid, 0)
        else:
            out = self.kernel(q,k,v,attention_mask)
        return out.transpose(1,2).reshape(b,n,c)
