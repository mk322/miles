"""Tree flash attention (``--sft-tree-attention``) against a dense masked reference.

The reference materializes the tree mask (causal inside every segment, plus each branch's rows
seeing its trunk prefix) and runs softmax attention per head in fp64, with the backward written
out by hand. The kernel runs in bf16/fp16, so its error is judged against the error plain
flash-attention makes on the same problem *expanded* into standalone per-branch sequences: the
tree must be as accurate as the expanded path, not merely "close".
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=1
)

import math
import random

import pytest
import torch
from flash_attn import flash_attn_varlen_func

from miles.backends.megatron_utils.tree_attention import TreeAttentionLayout, tree_flash_attention

DEVICE = "cuda"


def _random_layout(rng: random.Random, trees: list[tuple[int, list[tuple[int, int]]]], pad: int):
    """``trees``: per tree ``(segment0_len, [(branch_len, prefix_len), ...])``; returns host cu + cross."""
    cu, cross = [0], []
    for seg0, branches in trees:
        seg0_start = cu[-1]
        cu.append(seg0_start + seg0)
        for blen, prefix in branches:
            qs = cu[-1]
            cu.append(qs + blen)
            cross.append((qs, qs + blen, seg0_start, seg0_start + prefix))
    if pad:
        cu.append(cu[-1] + pad)
    return tuple(cu), tuple(cross)


def _random_trees(rng: random.Random, n_trees: int, max_trunk: int, max_branches: int, max_blen: int):
    trees = []
    for _ in range(n_trees):
        trunk = rng.randint(4, max_trunk)
        last_target = rng.randint(1, max_blen)
        prefixes = sorted(rng.randint(2, trunk - 1) for _ in range(rng.randint(0, max_branches)))
        trees.append((trunk + last_target, [(rng.randint(1, max_blen), p) for p in prefixes]))
    return trees


def _dense_mask(cu: tuple[int, ...], cross, total: int) -> torch.Tensor:
    mask = torch.zeros(total, total, dtype=torch.bool)
    for a, b in zip(cu[:-1], cu[1:], strict=True):
        mask[a:b, a:b] = torch.ones(b - a, b - a, dtype=torch.bool).tril()
    for qs, qe, ks, ke in cross:
        mask[qs:qe, ks:ke] = True
    return mask.to(DEVICE)


def _reference(q, k, v, dout, mask, scale):
    """fp64 masked attention and its gradients, one head at a time (GQA by index)."""
    T, hq, d = q.shape
    hk = k.shape[1]
    group = hq // hk
    q64, k64, v64, do64 = (x.double() for x in (q, k, v, dout))
    out = torch.empty_like(q64)
    dq, dk, dv = torch.zeros_like(q64), torch.zeros_like(k64), torch.zeros_like(v64)
    for h in range(hq):
        kh = h // group
        s = (q64[:, h] @ k64[:, kh].T) * scale
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, h] = p @ v64[:, kh]
        dv[:, kh] += p.T @ do64[:, h]
        dp = do64[:, h] @ v64[:, kh].T
        ds = p * (dp - (dp * p).sum(-1, keepdim=True))
        dq[:, h] = ds @ k64[:, kh] * scale
        dk[:, kh] += ds.T @ q64[:, h] * scale
    return out, dq, dk, dv


def _expanded_flash(q, k, v, dout, cu, cross, scale):
    """The same problem as standalone sequences (each branch with its prefix copied in front), via flash-attn.

    Returns output and grads mapped back onto the packed rows: what training the expanded samples would give.
    """
    seqs, rows = [], []  # rows: per standalone sequence, (packed row indices of its tokens, which are queries)
    cross_by_q = {qs: (qs, qe, ks, ke) for qs, qe, ks, ke in cross}
    for a, b in zip(cu[:-1], cu[1:], strict=True):
        if a in cross_by_q:
            _, _, ks, ke = cross_by_q[a]
            idx = list(range(ks, ke)) + list(range(a, b))
            seqs.append((idx, ke - ks))
        else:
            seqs.append((list(range(a, b)), 0))
    index = torch.tensor([i for idx, _ in seqs for i in idx], device=DEVICE)
    qx, kx, vx = (x[index].detach().clone().requires_grad_(True) for x in (q, k, v))
    lens = [len(idx) for idx, _ in seqs]
    cux = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEVICE)
    outx = flash_attn_varlen_func(qx, kx, vx, cux, cux, max(lens), max(lens), softmax_scale=scale, causal=True)
    # only the rows that are queries in the tree carry loss (a copied prefix row is the trunk's own business)
    is_query = torch.cat(
        [torch.tensor([False] * n_prefix + [True] * (len(idx) - n_prefix)) for idx, n_prefix in seqs]
    ).to(DEVICE)
    doutx = dout[index] * is_query[:, None, None]
    outx.backward(doutx)
    out = torch.empty_like(q)
    out[index[is_query]] = outx[is_query].detach()
    grads = []
    for gx, like in ((qx.grad, q), (kx.grad, k), (vx.grad, v)):
        g = torch.zeros(like.shape, dtype=torch.float64, device=DEVICE)
        g.index_add_(0, index, gx.double())
        grads.append(g)
    return out, *grads


def _run(cu, cross, hq, hk, d, dtype, seed=0):
    torch.manual_seed(seed)
    T = cu[-1]
    q = torch.randn(T, hq, d, device=DEVICE, dtype=dtype)
    k = torch.randn(T, hk, d, device=DEVICE, dtype=dtype)
    v = torch.randn(T, hk, d, device=DEVICE, dtype=dtype)
    dout = torch.randn(T, hq, d, device=DEVICE, dtype=dtype)
    scale = 1 / math.sqrt(d)
    layout = TreeAttentionLayout.build(cu, cross, DEVICE)

    qt, kt, vt = (x.clone().requires_grad_(True) for x in (q, k, v))
    out = tree_flash_attention(qt, kt, vt, layout, softmax_scale=scale)
    out.backward(dout)

    ref = _reference(q, k, v, dout, _dense_mask(cu, cross, T), scale)
    expanded = _expanded_flash(q, k, v, dout, cu, cross, scale)
    tree = (out.detach(), qt.grad, kt.grad, vt.grad)
    return tree, expanded, ref


def _assert_as_accurate_as_expanded(tree, expanded, ref, slack=2.0):
    for name, t, e, r in zip(("out", "dq", "dk", "dv"), tree, expanded, ref, strict=True):
        err_tree = (t.double() - r).abs().max().item()
        err_expanded = (e.double() - r).abs().max().item()
        scale = r.abs().max().item()
        # slack covers one extra bf16 rounding of the merged partials; the floor covers err_expanded == 0
        assert err_tree <= slack * err_expanded + 1e-3 * scale, (
            f"{name}: tree err {err_tree:.3e} vs expanded flash err {err_expanded:.3e} (|ref| {scale:.3e})"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hq,hk,d", [(32, 8, 128), (4, 4, 64), (6, 2, 128)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_trees_match_dense_reference(dtype, hq, hk, d, seed):
    rng = random.Random(seed)
    trees = _random_trees(rng, n_trees=rng.randint(1, 4), max_trunk=700, max_branches=6, max_blen=300)
    cu, cross = _random_layout(rng, trees, pad=rng.choice([0, 7, 128]))
    _assert_as_accurate_as_expanded(*_run(cu, cross, hq, hk, d, dtype, seed=seed))


@pytest.mark.parametrize("hq,hk,d", [(32, 8, 128), (8, 2, 128), (6, 2, 64)])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_forward_is_bitwise_the_expanded_samples(hq, hk, d, seed):
    """Every row (trunk, inline target, branch) equals its standalone sample's row bit for bit."""
    rng = random.Random(100 + seed)
    trees = _random_trees(rng, n_trees=rng.randint(1, 4), max_trunk=3000, max_branches=6, max_blen=700)
    cu, cross = _random_layout(rng, trees, pad=rng.choice([0, 33]))
    torch.manual_seed(seed)
    T = cu[-1]
    q = torch.randn(T, hq, d, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(T, hk, d, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(T, hk, d, device=DEVICE, dtype=torch.bfloat16)
    scale = 1 / math.sqrt(d)
    out = tree_flash_attention(q, k, v, TreeAttentionLayout.build(cu, cross, DEVICE), softmax_scale=scale)
    expanded_out, *_ = _expanded_flash(q, k, v, torch.zeros_like(q), cu, cross, scale)
    assert torch.equal(out, expanded_out)


def test_long_tree_qwen3_heads():
    """A 16K-token tree with Qwen3-8B's head layout (32 q heads, 8 kv heads, head_dim 128)."""
    rng = random.Random(7)
    trunk = 9000
    prefixes = sorted(rng.sample(range(500, trunk - 1), 10))
    trees = [(trunk + 900, [(rng.randint(200, 900), p) for p in prefixes])]
    cu, cross = _random_layout(rng, trees, pad=64)
    _assert_as_accurate_as_expanded(*_run(cu, cross, 32, 8, 128, torch.bfloat16))


def test_no_branches_is_bitwise_plain_flash():
    """Without branches the tree kernel is exactly flash-attn's causal varlen call, forward and backward."""
    cu = (0, 300, 1000, 1001, 1500)
    torch.manual_seed(0)
    q = torch.randn(1500, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1500, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1500, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    layout = TreeAttentionLayout.build(cu, (), DEVICE)
    a = [x.clone().requires_grad_(True) for x in (q, k, v)]
    b = [x.clone().requires_grad_(True) for x in (q, k, v)]
    out_a = tree_flash_attention(*a, layout, softmax_scale=0.1, deterministic=True)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    out_b = flash_attn_varlen_func(*b, cu_t, cu_t, 700, 700, softmax_scale=0.1, causal=True, deterministic=True)
    out_a.backward(dout)
    out_b.backward(dout)
    assert torch.equal(out_a, out_b)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x.grad, y.grad)


def test_branch_rows_do_not_leak_into_trunk_or_siblings():
    """Perturbing a branch's q/k/v changes only that branch's outputs and gradients wrt the trunk prefix."""
    cu, cross = (0, 64, 80, 100), ((64, 80, 0, 20), (80, 100, 0, 40))
    torch.manual_seed(0)
    q = torch.randn(100, 4, 64, device=DEVICE, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    layout = TreeAttentionLayout.build(cu, cross, DEVICE)
    base = tree_flash_attention(q, k, v, layout, softmax_scale=0.125)
    k2, v2 = k.clone(), v.clone()
    k2[64:80] += 1.0
    v2[64:80] += 1.0
    pert = tree_flash_attention(q, k2, v2, layout, softmax_scale=0.125)
    assert torch.equal(base[:64], pert[:64]), "trunk saw a branch"
    assert torch.equal(base[80:], pert[80:]), "a branch saw its sibling"
    assert not torch.equal(base[64:80], pert[64:80])
    # keys past a branch's prefix are invisible to it
    k3 = k.clone()
    k3[20:64] += 1.0
    pert = tree_flash_attention(q, k3, v, layout, softmax_scale=0.125)
    assert torch.equal(base[64:80], pert[64:80])
    assert not torch.equal(base[80:], pert[80:])


def test_layout_rejects_malformed_cross():
    with pytest.raises(AssertionError):
        TreeAttentionLayout.build((0, 10, 20), ((10, 20, 0, 15),), DEVICE)  # prefix reaches into the branch
    with pytest.raises(AssertionError):
        TreeAttentionLayout.build((0, 10, 20), ((12, 20, 0, 5),), DEVICE)  # rows are not a segment


@pytest.mark.parametrize("rope_fusion", [False, True])
def test_rope_at_positions_is_bitwise_megatron_thd_rope(nccl_world, rope_fusion):
    """Tree RoPE at explicit positions == Megatron's THD RoPE (positions restart per sequence), bit for bit."""
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles.backends.megatron_utils.tree_attention import apply_rope_at_positions

    parallel_state.initialize_model_parallel(1)
    try:
        config = TransformerConfig(
            num_layers=1, hidden_size=1024, num_attention_heads=8, kv_channels=128, apply_rope_fusion=rope_fusion
        )
        rotary = RotaryEmbedding(kv_channels=128, rotary_percent=1.0, rotary_base=1000000)
        lens = [300, 5000, 1]
        cu = torch.tensor([0, 300, 5300, 5301], dtype=torch.int32, device=DEVICE)
        torch.manual_seed(0)
        t = torch.randn(sum(lens), 8, 128, device=DEVICE, dtype=torch.bfloat16)
        cp = parallel_state.get_context_parallel_group()
        emb = rotary(5000, packed_seq=True)
        theirs = apply_rotary_pos_emb(t, emb, config=config, cu_seqlens=cu, cp_group=cp, max_seqlen=5000)
        positions = torch.cat([torch.arange(n) for n in lens]).to(DEVICE)
        assert torch.equal(apply_rope_at_positions(t, rotary, positions, 4999, config), theirs)
        # a branch starting mid-sequence rotates exactly like those positions inside the full sequence
        branch = apply_rope_at_positions(t[4300:4310], rotary, torch.arange(4000, 4010, device=DEVICE), 4999, config)
        assert torch.equal(branch, theirs[4300:4310])
    finally:
        parallel_state.destroy_model_parallel()


