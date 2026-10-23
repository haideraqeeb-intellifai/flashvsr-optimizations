"""Per-layer/head retention budgets for the original BF16 BSA kernel."""

import json
import math
from pathlib import Path

import torch


POLICY = Path(__file__).with_name("head-policy.json")


def _headwise_mask(batch_size, nheads, seqlen, q_w, k_w, local_attn_mask, rates):
    if batch_size != 1 or len(rates) != nheads:
        raise ValueError("Calibrated FlashVSR attention requires batch 1 and one budget per head")
    q = q_w.mean(1).reshape(-1, nheads, q_w.shape[-1] // nheads).permute(1, 0, 2)
    k = k_w.mean(1).reshape(-1, nheads, k_w.shape[-1] // nheads).permute(1, 0, 2)
    scores = torch.einsum("hld,hmd->hlm", q, k) / math.sqrt(q.shape[-1])
    eligible = local_attn_mask.repeat(
        scores.shape[1] // local_attn_mask.shape[0],
        scores.shape[2] // local_attn_mask.shape[1],
    )
    flat = scores.float().masked_fill(~eligible, -float("inf")).softmax(-1).reshape(nheads * seqlen, -1)
    counts = [min(flat.shape[1] - 1, max(1, math.ceil(flat.shape[1] * rate / 100)))
              for rate in rates for _ in range(seqlen)]
    ranked = torch.topk(flat, max(counts) + 1, dim=-1).values
    threshold = ranked.gather(1, torch.tensor(counts, device=q.device, dtype=torch.long)[:, None])
    return (flat > threshold).reshape(1, nheads, scores.shape[1], scores.shape[2])


def install_headwise_attention(pipe):
    policy = json.loads(POLICY.read_text())
    rates = policy["rates_percent"]
    blocks = pipe.dit.blocks
    if len(rates) != len(blocks) or any(len(row) != 12 for row in rates):
        raise RuntimeError("Head policy must have one 12-head budget row per FlashVSR block")
    if len(blocks) != 30 or blocks[0].self_attn.num_heads != 12:
        raise RuntimeError("Calibrated head policy requires the 30-layer, 12-head FlashVSR v1.1 model")
    dit = __import__("diffsynth.models.wan_video_dit", fromlist=["generate_draft_block_mask"])
    state = {"layer": 0}

    @torch.no_grad()
    def generate(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None):
        layer = state["layer"] % len(rates)
        state["layer"] += 1
        if local_attn_mask is None:
            raise ValueError("FlashVSR sparse attention requires its local attention mask")
        return _headwise_mask(batch_size, nheads, seqlen, q_w, k_w, local_attn_mask, rates[layer])

    dit.generate_draft_block_mask = generate
    return {
        "layers": len(rates), "heads_per_layer": 12,
        "mean_retention_percent": sum(map(sum, rates)) / (len(rates) * 12),
        "retention_percent": rates,
        "kernel": "original BF16 block-sparse attention",
    }
