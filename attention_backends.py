"""Explicit attention-only adapters. Never silently fall back to another backend."""
from collections import Counter
import importlib
from pathlib import Path
import sys
import json
import os
import subprocess
import time
import math

ROOT = Path(__file__).resolve().parent
BACKENDS = ('baseline', 'sparse_sage', 'sage22_dense', 'sage3_dense', 'jev_sparse', 'jev_replay')


class AttentionBackend:
    def __init__(self, name):
        import torch
        from diffsynth.models import wan_video_dit as dit
        self.name = name
        self.calls = Counter()
        self.shapes = Counter()
        self.original = dit.flash_attention
        # Installing Sage must not change the baseline/cross-attention dispatcher.
        dit.FLASH_ATTN_3_AVAILABLE = False
        dit.FLASH_ATTN_2_AVAILABLE = False
        dit.SAGE_ATTN_AVAILABLE = False
        self.torch = torch
        if name == 'sparse_sage':
            sys.path.insert(0, str(ROOT / 'vendor/ComfyUI-FlashVSR_Ultra_Fast/src/models'))
            self.kernel = importlib.import_module('sparse_sage.core').sparse_sageattn
        elif name == 'sage22_dense':
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            self.kernel = lambda q,k,v: sageattn_qk_int8_pv_fp8_cuda(
                q,k,v, tensor_layout='HND', is_causal=False, pv_accum_dtype='fp32+fp16')
        elif name == 'sage3_dense':
            from sageattn3 import sageattn3_blackwell
            self.kernel = sageattn3_blackwell
        elif name not in ('baseline','jev_sparse','jev_replay'):
            raise ValueError(name)
        dit.flash_attention = self.forward
        self.decisions = None
        self.rates = None
        if name in ('jev_sparse','jev_replay'):
            old_mask = dit.generate_draft_block_mask
            def jev_mask(batch_size,nheads,seqlen,q_w,k_w,topk=10,local_attn_mask=None):
                if self.rates is not None:
                    layer = self.calls['self'] % 30
                    pairs = (q_w.shape[0] // seqlen) * k_w.shape[0]
                    topk = max(1, math.ceil(pairs * self.rates[layer] / 100))
                return old_mask(batch_size,nheads,seqlen,q_w,k_w,topk,local_attn_mask)
            dit.generate_draft_block_mask = jev_mask

    def before_inference(self, case):
        if self.name == 'jev_replay':
            path = ROOT/'outputs/attention/jev_sparse'/case['case']/'metrics.json'
            saved = json.loads(path.read_text())
            assert saved['case'] == case['case']
            self.rates = list(saved['attention_backend']['jev']['applied_keep_percent'])
            assert len(self.rates) == 30 and all(r in (1,3,5,10) for r in self.rates)
            self.decisions = {'applied_keep_percent':self.rates,'api_wall_seconds':0,
                              'replay_source':str(path)}
            return
        if self.name != 'jev_sparse':
            return
        pilot = json.loads((ROOT/'outputs/attention/baseline/pilot-residuals.json').read_text())
        state = {'task':'4x video super-resolution, talking people', 'case_label':case['label'],
                 'num_layers':30, 'num_denoising_steps':1, 'initialization':True,
                 'historical_baseline_residuals':pilot,
                 'limitations':'Pilot from first 49 frames of female_default; no current-run statistics. '
                 'FlashVSR has no generated audio. Original source audio is copied. This is a FlashVSR '
                 'adaptation, not MiniMax H3 reproduction. Keep rate is fraction of block pairs per '
                 'head and temporal query group. Existing local mask remains. All model layers execute.',
                 'protection':'confidence below 0.3 becomes 10%; API failures abort, no silent fallback'}
        worker = os.environ.get('FLASHVSR_JEV_WORKER')
        if not worker:
            raise RuntimeError('Set FLASHVSR_JEV_WORKER to use the optional Jev backend')
        env = dict(os.environ)
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, worker], input=json.dumps(state),
                              text=True, capture_output=True, env=env, timeout=35)
        elapsed = time.perf_counter()-start
        if proc.returncode:
            raise RuntimeError('Jev decision failed; inspect worker access, no fallback was used')
        result = json.loads(proc.stdout)
        rates = []
        for i in range(30):
            d = result['decisions'][str(i)]
            rate = int(d['choice'])
            if rate not in (1,3,5,10):
                raise ValueError('Invalid Jev rate')
            rates.append(10 if d['confidence'] < .3 else rate)
        self.rates = rates
        self.decisions = {'request':state,'response':result,'applied_keep_percent':rates,'api_wall_seconds':elapsed}

    def reset(self):
        self.calls.clear()
        self.shapes.clear()

    def metadata(self):
        return {'backend': self.name, 'calls': dict(self.calls), 'shapes': dict(self.shapes),
                'self_attention_mask_preserved': self.name in ('baseline', 'sparse_sage'),
                'cross_attention': 'unchanged torch SDPA',
                'jev':self.decisions,
                'dense_note': 'Dense Sage replaces masked self-attention; original selector cost remains.'}

    def forward(self, q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
        self.calls['self' if attention_mask is not None else 'cross'] += 1
        if attention_mask is None or self.name in ('baseline','jev_sparse','jev_replay'):
            return self.original(q,k,v,num_heads,compatibility_mode,attention_mask,return_KV)
        self.shapes[str((tuple(q.shape),tuple(k.shape),num_heads))] += 1
        b, n, c = q.shape
        q,k,v = [x.reshape(b,-1,num_heads,c//num_heads).transpose(1,2).contiguous() for x in (q,k,v)]
        if self.name == 'sparse_sage':
            # Upstream baseline uses 128x128 blocks; Sparse Sage uses 128x64.
            # Duplicate each key-block flag to preserve EXACTLY the same token mask.
            mask = attention_mask.repeat_interleave(2,dim=-1).to(self.torch.int8).contiguous()
            out = self.kernel(q,k,v,mask_id=mask,is_causal=False,tensor_layout='HND')
        else:
            # Sage3 centers K in place, so pass a private copy to protect the KV cache.
            out = self.kernel(q,k.clone() if self.name == 'sage3_dense' else k,v)
        return out.transpose(1,2).reshape(b,n,c)
