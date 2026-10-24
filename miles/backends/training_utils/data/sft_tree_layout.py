"""Host-side layout of packed SFT trees for one micro-batch (see ``miles.rollout.sft_tree``).

A sample's ``tree_branches`` is ``[(branch_len, prefix_len), ...]``: the sample ends with those
branches in order, and everything before them is segment 0 (trunk + inline target). Branch ``i``
attends to the first ``prefix_len`` tokens of its own sample's segment 0 and is positioned from
``prefix_len``. A sample without branches is one ordinary causal sequence.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TreeBatchLayout:
    cu_seqlens: tuple[int, ...]  # causal segment boundaries over the packed micro-batch (incl. padding)
    cross: tuple[tuple[int, int, int, int], ...]  # (q_start, q_end, k_start, k_end) per branch
    positions: tuple[int, ...]  # RoPE position of every packed token (padding counts from 0)


def build_tree_batch_layout(
    total_lengths: Sequence[int], tree_branches: Sequence[Sequence[Sequence[int]] | None], pad: int
) -> TreeBatchLayout:
    """Layout of samples packed back to back (CP 1), followed by ``pad`` padding tokens."""
    cu, cross, positions = [0], [], []
    for total, branches in zip(total_lengths, tree_branches, strict=True):
        branches = [tuple(b) for b in (branches or ())]
        seg0 = total - sum(blen for blen, _ in branches)
        assert seg0 > 0 and all(blen > 0 for blen, _ in branches), f"bad tree {total=} {branches=}"
        start = cu[-1]
        cu.append(start + seg0)
        positions.extend(range(seg0))
        for blen, prefix in branches:
            assert 0 < prefix < seg0, f"branch prefix {prefix} outside segment 0 of length {seg0}"
            q_start = cu[-1]
            cu.append(q_start + blen)
            cross.append((q_start, q_start + blen, start, start + prefix))
            positions.extend(range(prefix, prefix + blen))
    if pad:
        cu.append(cu[-1] + pad)
        positions.extend(range(pad))
    return TreeBatchLayout(cu_seqlens=tuple(cu), cross=tuple(cross), positions=tuple(positions))


def visible_keys(layout: TreeBatchLayout, row: int) -> list[int]:
    """The packed rows ``row`` attends to, in order: the tree mask, spelled out (for tests and debugging)."""
    seg = max(i for i, b in enumerate(layout.cu_seqlens[:-1]) if b <= row)
    seg_start = layout.cu_seqlens[seg]
    keys = []
    for q_start, q_end, k_start, k_end in layout.cross:
        if q_start <= row < q_end:
            keys.extend(range(k_start, k_end))
    return keys + list(range(seg_start, row + 1))
