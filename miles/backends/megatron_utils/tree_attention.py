"""Tree attention for SFT on multi-turn traces (``--sft-tree-attention``).

A packed tree sample is a run of *segments*. Segment 0 is the trunk (the inference prompt of the
tree's last turn) followed by that turn's target; every later segment is one earlier turn's
branch. Attention inside every segment is causal, and a branch additionally attends to all of a
prefix of its tree's trunk (the keys ``[k_start, k_end)``): exactly the context the turn has at
inference. Trunk tokens never see branches, and branches never see each other.

The kernel is flash-attention 2 twice over, with no gathered or copied keys:

* one causal varlen call over every segment of the micro-batch (``cu_seqlens``);
* one non-causal call per branch, of its queries against a contiguous trunk prefix (a view).

The two partial softmaxes of a branch row are merged through their log-sum-exps. The backward
pass hands both flash backward calls the *merged* output and log-sum-exp, which is what makes the
gradient of each key subset the gradient of the full softmax (the ring-attention identity).

RoPE: the tree's position ids are not a function of ``cu_seqlens`` (a branch restarts at its attach
point, not at 0), so in tree mode the model-level RoPE is switched off and ``TreeCoreAttention``
rotates q and k itself at explicit per-token position ids, through Megatron's own RoPE function
(bit-identical to the existing THD RoPE at the same positions, fused or not).
"""

import logging
import math
import os
from dataclasses import dataclass, field

import torch
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward, _flash_attn_varlen_forward
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention

from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens
from miles.backends.training_utils.data.sft_tree_layout import TreeBatchLayout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TreeAttentionLayout:
    """Where every query of a packed micro-batch may look.

    ``cu_seqlens`` (device int32) bounds the causal segments. ``cross`` holds, per branch, the host
    ints ``(q_start, q_end, k_start, k_end)``: rows ``[q_start, q_end)`` also see keys
    ``[k_start, k_end)``, all of them, on top of their causal segment. ``cross_cu_seqlens`` is the
    device tensor ``[[0, q_end - q_start], [0, k_end - k_start]]`` per branch, built once.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
    cross: tuple[tuple[int, int, int, int], ...]
    cross_cu_seqlens: torch.Tensor | None

    @staticmethod
    def build(
        cu_seqlens_host: tuple[int, ...], cross: tuple[tuple[int, int, int, int], ...], device
    ) -> "TreeAttentionLayout":
        seg_lens = [b - a for a, b in zip(cu_seqlens_host[:-1], cu_seqlens_host[1:], strict=True)]
        _validate_cross(cu_seqlens_host, cross)
        cross_cu = None
        if cross:
            cross_cu = torch.tensor(
                [[[0, qe - qs], [0, ke - ks]] for qs, qe, ks, ke in cross], dtype=torch.int32, device=device
            )
        return TreeAttentionLayout(
            cu_seqlens=torch.tensor(cu_seqlens_host, dtype=torch.int32, device=device),
            max_seqlen=max(seg_lens),
            cross=tuple(cross),
            cross_cu_seqlens=cross_cu,
        )


def _validate_cross(cu_seqlens_host: tuple[int, ...], cross: tuple[tuple[int, int, int, int], ...]) -> None:
    starts = set(cu_seqlens_host[:-1])
    for qs, qe, ks, ke in cross:
        # a branch is a whole segment, and it looks strictly backwards into another one
        assert qs in starts and qe in cu_seqlens_host, f"cross rows [{qs}, {qe}) are not one segment"
        assert ks in starts and 0 < ke - ks and ke <= qs, f"cross keys [{ks}, {ke}) for rows [{qs}, {qe})"


@dataclass
class TreePackedSeqParams(PackedSeqParamsWithHostCuSeqlens):
    """THD packed params plus the tree layout and the per-token RoPE positions.

    ``cu_seqlens_q``/``cu_seqlens_kv`` are the causal segments, not the samples.
    """

    tree_layout: TreeAttentionLayout = field(kw_only=True)
    position_ids: torch.Tensor = field(kw_only=True)
    max_position: int = field(kw_only=True)


def _fwd(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal):
    out, lse, _, _ = _flash_attn_varlen_forward(
        q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal
    )
    return out, lse


def _bwd(dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, softmax_scale, causal, deterministic):
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_q,
        cu_k,
        max_q,
        max_k,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        deterministic,
    )
    return dq, dk, dv


def _merge_into(out: torch.Tensor, lse: torch.Tensor, rows: slice, out2: torch.Tensor, lse2: torch.Tensor) -> None:
    """Merge a second partial softmax ``(out2, lse2)`` into ``out[rows]``/``lse[:, rows]`` in place (fp32 math)."""
    lse1 = lse[:, rows]
    merged = torch.logaddexp(lse1, lse2)
    w1 = torch.exp(lse1 - merged).transpose(0, 1).unsqueeze(-1)  # [rows, heads, 1]
    w2 = torch.exp(lse2 - merged).transpose(0, 1).unsqueeze(-1)
    out[rows] = (out[rows].float() * w1 + out2.float() * w2).to(out.dtype)
    lse[:, rows] = merged


class _TreeFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, layout: TreeAttentionLayout, softmax_scale: float, deterministic: bool):
        cu, max_len = layout.cu_seqlens, layout.max_seqlen
        out, lse = _fwd(q, k, v, cu, cu, max_len, max_len, softmax_scale, causal=True)
        for i, (qs, qe, ks, ke) in enumerate(layout.cross):
            cu_q, cu_k = layout.cross_cu_seqlens[i, 0], layout.cross_cu_seqlens[i, 1]
            out2, lse2 = _fwd(q[qs:qe], k[ks:ke], v[ks:ke], cu_q, cu_k, qe - qs, ke - ks, softmax_scale, causal=False)
            _merge_into(out, lse, slice(qs, qe), out2, lse2)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.layout, ctx.softmax_scale, ctx.deterministic = layout, softmax_scale, deterministic
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        layout, scale, det = ctx.layout, ctx.softmax_scale, ctx.deterministic
        dout = dout.contiguous()
        cu, max_len = layout.cu_seqlens, layout.max_seqlen
        dq, dk, dv = _bwd(dout, q, k, v, out, lse, cu, cu, max_len, max_len, scale, True, det)
        if not layout.cross:
            return dq, dk, dv, None, None, None
        # A trunk key collects gradient from every branch that sees it: sum the parts in fp32 and
        # round once, as a single kernel would, instead of rounding the running sum per branch.
        dq, dk, dv = dq.float(), dk.float(), dv.float()
        for i, (qs, qe, ks, ke) in enumerate(layout.cross):
            cu_q, cu_k = layout.cross_cu_seqlens[i, 0], layout.cross_cu_seqlens[i, 1]
            dq2, dk2, dv2 = _bwd(
                dout[qs:qe],
                q[qs:qe],
                k[ks:ke],
                v[ks:ke],
                out[qs:qe],
                lse[:, qs:qe].contiguous(),
                cu_q,
                cu_k,
                qe - qs,
                ke - ks,
                scale,
                False,
                det,
            )
            dq[qs:qe] += dq2
            dk[ks:ke] += dk2
            dv[ks:ke] += dv2
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def tree_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: TreeAttentionLayout,
    *,
    softmax_scale: float,
    deterministic: bool = False,
) -> torch.Tensor:
    """Tree attention over THD tensors ``q [T, Hq, D]``, ``k``/``v [T, Hkv, D]`` (GQA: ``Hq % Hkv == 0``)."""
    assert q.dim() == 3 and k.shape == v.shape and q.shape[0] == k.shape[0], f"{q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    return _TreeFlashAttention.apply(q, k, v, layout, softmax_scale, deterministic)


def apply_rope_at_positions(
    t: torch.Tensor, rotary_emb: torch.nn.Module, positions: torch.Tensor, max_position: int, config
) -> torch.Tensor:
    """RoPE of THD ``t [T, H, D]`` at explicit per-token ``positions``.

    The angles of every token are gathered from the model's ``RotaryEmbedding`` and applied through
    Megatron's own ``apply_rotary_pos_emb`` in its sbhd form (``s = T``, ``b = 1``), fused or not as
    ``config.apply_rope_fusion`` says: the same arithmetic, bit for bit, as the THD RoPE of the
    existing path at the same positions.
    """
    freqs = rotary_emb.get_emb(max_position + 1)[positions]  # [T, 1, 1, D_rot]
    return apply_rotary_pos_emb(t.unsqueeze(1), freqs, config=config).squeeze(1)


class TreeCoreAttention(torch.nn.Module):
    """Drop-in for a self-attention's ``core_attention`` that applies RoPE and tree attention.

    ``rotary_emb`` is the model's ``RotaryEmbedding``: ``rotary_emb.get_emb(n)`` returns the angles
    of positions ``[0, n)`` as ``[n, 1, 1, D_rot]``, before any CP sharding.
    """

    def __init__(self, *, config, rotary_emb: torch.nn.Module, softmax_scale: float):
        super().__init__()
        # plain attributes, not submodules: the model owns them, and neither has parameters
        object.__setattr__(self, "_rotary_emb", rotary_emb)
        object.__setattr__(self, "_config", config)
        self.softmax_scale = softmax_scale
        # flash-attn's backward is deterministic only on request; ask exactly when TE would
        self.deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> torch.Tensor:
        assert isinstance(packed_seq_params, TreePackedSeqParams), (
            "--sft-tree-attention needs every batch packed by get_batch with a tree layout, "
            f"got {type(packed_seq_params).__name__}"
        )
        assert attention_bias is None
        positions = packed_seq_params.position_ids
        assert query.shape[0] == positions.shape[0], f"{query.shape} vs {positions.shape}"
        max_position = packed_seq_params.max_position
        query = apply_rope_at_positions(query, self._rotary_emb, positions, max_position, self._config).contiguous()
        key = apply_rope_at_positions(key, self._rotary_emb, positions, max_position, self._config).contiguous()
        out = tree_flash_attention(
            query,
            key,
            value.contiguous(),
            packed_seq_params.tree_layout,
            softmax_scale=self.softmax_scale,
            deterministic=self.deterministic,
        )
        return out.reshape(out.shape[0], -1)


def get_tree_packed_seq_params(host_layout: TreeBatchLayout) -> TreePackedSeqParams:
    """Device-side packed params of one micro-batch's ``TreeBatchLayout`` (built by ``get_batch``)."""
    device = torch.cuda.current_device()
    layout = TreeAttentionLayout.build(host_layout.cu_seqlens, host_layout.cross, device)
    cu = layout.cu_seqlens
    return TreePackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=layout.max_seqlen,
        max_seqlen_kv=layout.max_seqlen,
        qkv_format="thd",
        cu_seqlens_host=host_layout.cu_seqlens,
        total_tokens=host_layout.cu_seqlens[-1],
        tree_layout=layout,
        position_ids=torch.tensor(host_layout.positions, dtype=torch.long, device=device),
        max_position=max(host_layout.positions),
    )


def install_tree_attention(model: torch.nn.Module) -> None:
    """Switch one GPTModel chunk to tree attention: every self-attention's core attention, and RoPE.

    The model-level RoPE is turned off (``position_embedding_type`` no longer ``'rope'``, so the
    model passes ``rotary_pos_emb=None`` down) and ``TreeCoreAttention`` applies it from the batch's
    explicit position ids instead, with the same ``RotaryEmbedding`` module.
    """
    assert model.position_embedding_type == "rope", f"tree attention needs RoPE, got {model.position_embedding_type}"
    config = model.config
    assert not config.multi_latent_attention and config.attention_dropout == 0.0
    rotary_emb = model.rotary_pos_emb
    softmax_scale = config.softmax_scale or 1.0 / math.sqrt(config.kv_channels)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            assert module.rotary_pos_emb is None, f"{name}: per-layer RoPE is not supported"
            assert module._yarn_concentration_factor == 1.0, f"{name}: RoPE mscale is not supported"
            module.core_attention = TreeCoreAttention(
                config=module.config, rotary_emb=rotary_emb, softmax_scale=softmax_scale
            )
            count += 1
    assert count == len(model.decoder.layers), f"replaced {count} attentions for {len(model.decoder.layers)} layers"
    model.position_embedding_type = "none"
    logger.info(f"sft tree attention: replaced core attention and RoPE in {count} layers")
