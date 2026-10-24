"""Benchmark: tree flash attention vs the existing causal path, fwd+bwd, at OTA-like shapes.

For a trace whose tree is ``trunk + inline target + branches`` (lengths from a real aligned row
distribution), compare per-layer attention time and peak memory of:

* ``tree``: tree_flash_attention on the tree (what --sft-tree-attention runs);
* ``verbatim``: one causal flash-attn sequence of the same total length (the cost of the current
  path on as many tokens);
* ``expanded``: flash-attn over the standalone per-turn samples (the exact objective done naively).

    CUDA_VISIBLE_DEVICES=<gpu> python tests/manual/sft_tree/bench_tree_attention.py
"""

import json
import math

import torch
from flash_attn import flash_attn_varlen_func

from miles.backends.megatron_utils.tree_attention import TreeAttentionLayout, tree_flash_attention


def _time(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**30


def _case(trunk, last_target, branches, hq, hk, d=128):
    """branches: [(target_len, prefix_len)]; returns timings (ms) and peak memory (GiB)."""
    seg0 = trunk + last_target
    cu, cross = [0, seg0], []
    for blen, prefix in branches:
        cu.append(cu[-1] + blen + 1)
        cross.append((cu[-2], cu[-1], 0, prefix))
    T = cu[-1]
    scale = 1 / math.sqrt(d)
    q = torch.randn(T, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(T, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(T, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layout = TreeAttentionLayout.build(tuple(cu), tuple(cross), "cuda")
    dout = torch.randn(T, hq, d, device="cuda", dtype=torch.bfloat16)

    def tree():
        tree_flash_attention(q, k, v, layout, softmax_scale=scale).backward(dout)

    cu_v = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    def verbatim():
        flash_attn_varlen_func(q, k, v, cu_v, cu_v, T, T, softmax_scale=scale, causal=True).backward(dout)

    lens = [seg0] + [prefix + blen + 1 for blen, prefix in branches]
    E = sum(lens)
    qe = torch.randn(E, hq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ke = torch.randn(E, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ve = torch.randn(E, hk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu_e = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    dout_e = torch.randn(E, hq, d, device="cuda", dtype=torch.bfloat16)

    def expanded():
        flash_attn_varlen_func(qe, ke, ve, cu_e, cu_e, max(lens), max(lens), softmax_scale=scale, causal=True).backward(
            dout_e
        )

    out = {"tokens_tree": T, "tokens_expanded": E}
    for name, fn in (("tree", tree), ("verbatim_same_tokens", verbatim), ("expanded", expanded)):
        ms, gib = _time(fn)
        out[name] = {"ms": round(ms, 2), "peak_gib": round(gib, 2)}
    return out


def main():
    torch.manual_seed(0)
    results = {}
    # a typical OTA tree: ~12 turns, prompts growing to ~25K, think-heavy targets of ~1-2K
    trunk, prefixes, targets = 25000, [], []
    p = 3000
    for _ in range(11):
        prefixes.append(p)
        targets.append(1600)
        p += 1900
    for heads, name in (((32, 8), "tp1"), ((16, 4), "tp2"), ((8, 2), "tp4")):
        results[name] = _case(trunk, 1500, list(zip(targets, prefixes, strict=True)), *heads)
        print(name, json.dumps(results[name]), flush=True)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
