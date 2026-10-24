"""Tree attention for SFT on multi-turn traces (``--sft-tree-attention``).

A packed tree sample is a run of *segments*. Segment 0 is the trunk (the inference prompt of the
tree's last turn) followed by that turn's target; every later segment is one earlier turn's
branch. Attention inside every segment is causal, and a branch additionally attends to all of a
prefix of its tree's trunk (the keys ``[k_start, k_end)``): exactly the context the turn has at
inference. Trunk tokens never see branches, and branches never see each other.

The kernel is flash-attention 2, arranged so that every row runs exactly the arithmetic it runs
in its standalone per-turn sample:

* one causal varlen call over every segment of the micro-batch (``cu_seqlens``) gives the trunk
  and inline-target rows (a segment 0 is literally the last turn's standalone sample);
* per branch, one causal call of its queries against its keys gathered as ``[trunk prefix;
  branch]`` (bottom-right aligned): the key sequence of the branch's standalone sample, so its rows
  are computed as in that sample, and overwrite what the segment call computed for them.

Backward: the segment call's backward sees zero output-gradient on branch rows (so those rows
contribute nothing to it: dS = P * (dP - D) = 0 exactly), and each branch's own call adds its dq
and its key gradients, scattered back to the trunk prefix and the branch, summed in fp32 and
rounded once. The gathered keys are rebuilt in backward, never saved.

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
    device tensor ``[[0, q_len], [0, k_len + q_len]]`` per branch (``q_len = q_end - q_start``,
    ``k_len = k_end - k_start``): the branch's queries and its ``[prefix; branch]`` keys, built once.
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
                [[[0, qe - qs], [0, (ke - ks) + (qe - qs)]] for qs, qe, ks, ke in cross],
                dtype=torch.int32,
                device=device,
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


def _branch_kv(k: torch.Tensor, v: torch.Tensor, branch: tuple[int, int, int, int]):
    """A branch's standalone key/value sequence: ``[trunk prefix; branch]``."""
    qs, qe, ks, ke = branch
    return torch.cat((k[ks:ke], k[qs:qe])), torch.cat((v[ks:ke], v[qs:qe]))


class _TreeFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, layout: TreeAttentionLayout, softmax_scale: float, deterministic: bool):
        cu, max_len = layout.cu_seqlens, layout.max_seqlen
        out, lse = _fwd(q, k, v, cu, cu, max_len, max_len, softmax_scale, causal=True)
        for i, branch in enumerate(layout.cross):
            qs, qe, ks, ke = branch
            kb, vb = _branch_kv(k, v, branch)
            cu_q, cu_k = layout.cross_cu_seqlens[i, 0], layout.cross_cu_seqlens[i, 1]
            out_b, lse_b = _fwd(q[qs:qe], kb, vb, cu_q, cu_k, qe - qs, kb.shape[0], softmax_scale, causal=True)
            out[qs:qe] = out_b
            lse[:, qs:qe] = lse_b
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.layout, ctx.softmax_scale, ctx.deterministic = layout, softmax_scale, deterministic
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        layout, scale, det = ctx.layout, ctx.softmax_scale, ctx.deterministic
        dout = dout.contiguous()
        cu, max_len = layout.cu_seqlens, layout.max_seqlen
        if not layout.cross:
            dq, dk, dv = _bwd(dout, q, k, v, out, lse, cu, cu, max_len, max_len, scale, True, det)
            return dq, dk, dv, None, None, None
        # Branch rows were not produced by the segment call: give them no gradient there. With
        # dO = 0 their dS is exactly 0 whatever out/lse hold, so they add nothing to dq, dk or dv.
        dout_segments = dout.clone()
        for qs, qe, _, _ in layout.cross:
            dout_segments[qs:qe] = 0
        dq, dk, dv = _bwd(dout_segments, q, k, v, out, lse, cu, cu, max_len, max_len, scale, True, det)
        # A trunk key collects gradient from its segment and from every branch that sees it: sum
        # the parts in fp32 and round once.
        dk, dv = dk.float(), dv.float()
        for i, branch in enumerate(layout.cross):
            qs, qe, ks, ke = branch
            kb, vb = _branch_kv(k, v, branch)
            cu_q, cu_k = layout.cross_cu_seqlens[i, 0], layout.cross_cu_seqlens[i, 1]
            dq_b, dk_b, dv_b = _bwd(
                dout[qs:qe],
                q[qs:qe],
                kb,
                vb,
                out[qs:qe],
                lse[:, qs:qe].contiguous(),
                cu_q,
                cu_k,
                qe - qs,
                kb.shape[0],
                scale,
                True,
                det,
            )
            prefix = ke - ks
            dq[qs:qe] = dq_b  # the segment call gave these rows exactly 0
            dk[ks:ke] += dk_b[:prefix]
            dk[qs:qe] += dk_b[prefix:]
            dv[ks:ke] += dv_b[:prefix]
            dv[qs:qe] += dv_b[prefix:]
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None


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


class TreeCoreAttention:
    """The ``forward`` of a self-attention's ``core_attention`` under tree attention: RoPE, then tree attention.

    It is installed as the forward of the existing core-attention module rather than replacing the
    module, so the model's parameters and state-dict keys (TE's ``_extra_state``) are exactly those
    of the existing path and checkpoints load either way. ``rotary_emb`` is the model's
    ``RotaryEmbedding``: ``rotary_emb.get_emb(n)`` returns the angles of positions ``[0, n)`` as
    ``[n, 1, 1, D_rot]``, before any CP sharding.
    """

    def __init__(self, *, config, rotary_emb: torch.nn.Module, softmax_scale: float):
        self._rotary_emb = rotary_emb
        self._config = config
        self.softmax_scale = softmax_scale
        # flash-attn's backward is deterministic only on request; ask exactly when TE would
        self.deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )

    def __call__(
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
            # nn.Module.__call__ dispatches to the instance attribute, hooks and all
            module.core_attention.forward = TreeCoreAttention(
                config=module.config, rotary_emb=rotary_emb, softmax_scale=softmax_scale
            )
            count += 1
    assert count == len(model.decoder.layers), f"replaced {count} attentions for {len(model.decoder.layers)} layers"
    model.position_embedding_type = "none"
    logger.info(f"sft tree attention: replaced core attention and RoPE in {count} layers")
