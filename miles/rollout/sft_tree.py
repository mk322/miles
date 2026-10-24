"""Pack a multi-turn SFT trace into inference-aligned trees (``--sft-tree-attention``).

At inference every assistant turn ``k`` is generated after the chat template's rendering of the
messages before it (``add_generation_prompt=True``); the stock Qwen3 template strips the
``<think>`` block of every earlier assistant turn there. The exact objective is therefore one
standalone sample per supervised turn: ``prompt_k = tokenize(render(messages[:i_k]))`` (unsupervised)
followed by ``target_k`` (supervised). This module packs consecutive turns into trees that train
exactly that objective at a fraction of the tokens.

A tree over turns ``k_1 < ... < k_n`` has the token layout::

    [ trunk = prompt_{k_n} ][ target_{k_n} ][ d_{k_1} target_{k_1} ] ... [ d_{k_{n-1}} target_{k_{n-1}} ]

* the trunk is the last turn's prompt; every ``prompt_k`` of the tree must be a token prefix of it
  (checked, never assumed: a template that re-renders history, a BPE merge across the generation
  prompt, ... starts a new tree instead);
* the last turn's target continues the trunk causally, so a one-turn tree *is* the standalone sample;
* every other turn is a *branch*: ``d_k`` repeats the trunk token ``prompt_k[-1]`` (the last token
  of the generation prompt) at its own position, and the branch attends to the trunk prefix
  ``prompt_k[:-1]`` plus itself, causally. ``d_k`` therefore computes exactly the hidden states of
  trunk position ``len(prompt_k) - 1`` and its logit predicts ``target_k[0]``, while the trunk's own
  next token there (the think-stripped history) is never a label. ``d_k`` carries loss mask 0.

Position ids: trunk and inline target count from 0; branch ``k`` counts from ``len(prompt_k) - 1``,
so every token sits at the position it has in its standalone sample.

The loss mask covers the response (everything after the trunk): 1 on every target token, 0 on
every ``d_k`` (whose own label would be the previous branch's last token). Training the tree is
the sum of the standalone per-turn samples' losses, token for token.

Length limits: a turn whose standalone sample exceeds ``max_context_len`` (the inference context)
is dropped whole, never truncated. Turns are split into consecutive sub-trees of at most
``max_tree_tokens`` tokens each (the per-GPU budget), choosing the split that minimizes total tokens;
each sub-tree re-materializes its own trunk.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Message keys this module understands. Anything else (``reasoning_content``, ``tool_calls``, ...)
# would change what the template renders or what the target is, so it is rejected, not dropped.
_MESSAGE_KEYS = frozenset({"role", "content", "loss"})
_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class TurnSample:
    """One supervised assistant turn as a standalone sample."""

    message_index: int
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.target_ids)


@dataclass(frozen=True)
class TreeSample:
    """A packed tree. ``branches`` is ``(branch_len, prefix_len)`` per branch, in token order after the inline target."""

    tokens: tuple[int, ...]
    response_length: int
    loss_mask: tuple[int, ...]
    branches: tuple[tuple[int, int], ...]
    turns: tuple[TurnSample, ...]  # inline (last) turn first, then the branches in order


@dataclass(frozen=True)
class TreeStats:
    num_turns: int  # supervised turns in the trace
    num_dropped_turns: int  # standalone sample longer than max_context_len
    dropped_target_tokens: int
    num_trees: int
    num_prefix_breaks: int  # a turn whose prompt is not a token prefix of the next turn's
    tree_tokens: int  # total tokens of all trees
    expanded_tokens: int  # total tokens of the standalone samples of the kept turns
    target_tokens: int  # supervised tokens kept


def validate_messages(messages: Sequence[dict]) -> None:
    """Reject anything the stock template would not render to exactly what inference sends."""
    assert messages, "empty conversation"
    for i, m in enumerate(messages):
        extra = set(m) - _MESSAGE_KEYS
        assert not extra, f"message {i}: unsupported keys {sorted(extra)} (only {sorted(_MESSAGE_KEYS)})"
        assert m.get("role") in _ROLES, f"message {i}: unsupported role {m.get('role')!r}"
        assert isinstance(m.get("content"), str), f"message {i}: content must be a string"
        if "loss" in m:
            assert m["role"] == "assistant", f"message {i}: `loss` is only meaningful on assistant turns"
            assert isinstance(m["loss"], bool), f"message {i}: `loss` must be a bool"
        if m["role"] == "system":
            # the template renders a non-first system message as a plain turn; inference never sends one
            assert i == 0, f"message {i}: system message must be first"


def _is_tool_response(content: str) -> bool:
    return content.startswith("<tool_response>") and content.endswith("</tool_response>")


def _assert_prompt_ends_with_query(messages: Sequence[dict], i: int) -> None:
    """A supervised turn's prompt must end with a real user query.

    The stock template's ``last_query_index`` is then the message right before the turn, so every
    earlier assistant turn is rendered think-stripped: the shape this packing (and the trunk prefix
    property) is meant for. A turn after a ``<tool_response>`` user, after another assistant, or with
    no prompt at all is rejected.
    """
    assert i > 0, f"message {i}: a supervised assistant turn needs a prompt"
    prev = messages[i - 1]
    assert prev["role"] == "user", f"message {i}: a supervised turn must follow a user message, not {prev['role']}"
    assert not _is_tool_response(prev["content"]), f"message {i}: follows a <tool_response> user message"


def build_turn_samples(
    tokenizer, messages: Sequence[dict], *, target_suffix: str, chat_template: str | None = None
) -> list[TurnSample]:
    """Every supervised assistant turn as ``(prompt_ids, target_ids)``.

    ``prompt_ids`` is the chat template's generation prompt for the messages before the turn,
    tokenized as one string (as vLLM does); ``target_ids`` is the raw content plus ``target_suffix``,
    tokenized on its own.
    """
    validate_messages(messages)
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    indices = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("loss", True)]
    for i in indices:
        _assert_prompt_ends_with_query(history, i)
    if not indices:
        return []
    prompts = [
        tokenizer.apply_chat_template(
            history[:i], tokenize=False, add_generation_prompt=True, chat_template=chat_template
        )
        for i in indices
    ]
    targets = [messages[i]["content"] + target_suffix for i in indices]
    encoded = tokenizer(prompts + targets, add_special_tokens=False)["input_ids"]
    n = len(indices)
    return [
        TurnSample(message_index=i, prompt_ids=tuple(p), target_ids=tuple(t))
        for i, p, t in zip(indices, encoded[:n], encoded[n:], strict=True)
    ]


def _is_prefix(short: tuple[int, ...], long: tuple[int, ...]) -> bool:
    return len(short) <= len(long) and long[: len(short)] == short


def _tree_tokens(turns: Sequence[TurnSample]) -> int:
    *branches, last = turns
    return last.length + sum(1 + len(t.target_ids) for t in branches)


def _plan_trees(turns: Sequence[TurnSample], max_tree_tokens: int) -> list[tuple[int, int]]:
    """Split ``turns`` into consecutive ``[start, end)`` groups minimizing total tree tokens.

    A group is feasible when it fits ``max_tree_tokens`` and every prompt in it is a token prefix
    of the group's last prompt. Ties prefer fewer trees. A single turn is always feasible here.
    """
    n = len(turns)
    inf = (float("inf"), 0)
    best: list[tuple[float, int]] = [(0, 0)] + [inf] * n  # best[e] = (tokens, trees) for turns[:e]
    choice = [0] * (n + 1)
    for end in range(1, n + 1):
        last = turns[end - 1]
        for start in range(end - 1, -1, -1):
            if not _is_prefix(turns[start].prompt_ids, last.prompt_ids):
                break  # a group must be contiguous, and this prompt already breaks it
            group = turns[start:end]
            size = _tree_tokens(group)
            if size > max_tree_tokens and end - start > 1:
                break  # growing the group further back only adds tokens
            cand = (best[start][0] + size, best[start][1] + 1)
            if cand < best[end]:
                best[end], choice[end] = cand, start
    groups, end = [], n
    while end > 0:
        groups.append((choice[end], end))
        end = choice[end]
    return groups[::-1]


def assemble_tree(turns: Sequence[TurnSample]) -> TreeSample:
    """Lay out one tree (see the module docstring); ``turns`` in conversation order."""
    *branch_turns, last = turns
    trunk = last.prompt_ids
    tokens = list(trunk) + list(last.target_ids)
    loss_mask = [1] * len(last.target_ids)
    branches = []
    for t in branch_turns:
        assert _is_prefix(t.prompt_ids, trunk), "a branch prompt must be a token prefix of the trunk"
        tokens += [trunk[len(t.prompt_ids) - 1]] + list(t.target_ids)
        loss_mask += [0] + [1] * len(t.target_ids)
        branches.append((1 + len(t.target_ids), len(t.prompt_ids) - 1))
    return TreeSample(
        tokens=tuple(tokens),
        response_length=len(tokens) - len(trunk),
        loss_mask=tuple(loss_mask),
        branches=tuple(branches),
        turns=(last, *branch_turns),
    )


def build_trees(
    turns: Sequence[TurnSample], *, max_context_len: int, max_tree_tokens: int
) -> tuple[list[TreeSample], TreeStats]:
    """Drop turns longer than ``max_context_len``, then pack the rest into trees of at most ``max_tree_tokens``."""
    assert max_tree_tokens >= max_context_len, "a kept turn must always fit one tree on its own"
    kept = [t for t in turns if t.length <= max_context_len]
    dropped = [t for t in turns if t.length > max_context_len]
    groups = _plan_trees(kept, max_tree_tokens)
    trees = [assemble_tree(kept[a:b]) for a, b in groups]
    breaks = sum(not _is_prefix(a.prompt_ids, b.prompt_ids) for a, b in zip(kept[:-1], kept[1:], strict=True))
    stats = TreeStats(
        num_turns=len(turns),
        num_dropped_turns=len(dropped),
        dropped_target_tokens=sum(len(t.target_ids) for t in dropped),
        num_trees=len(trees),
        num_prefix_breaks=breaks,
        tree_tokens=sum(len(t.tokens) for t in trees),
        expanded_tokens=sum(t.length for t in kept),
        target_tokens=sum(len(t.target_ids) for t in kept),
    )
    return trees, stats


def expand_tree(tree: TreeSample) -> list[tuple[list[int], list[int]]]:
    """The standalone ``(tokens, loss_mask over tokens)`` samples a tree stands for, in ``tree.turns`` order."""
    return [
        (list(t.prompt_ids) + list(t.target_ids), [0] * len(t.prompt_ids) + [1] * len(t.target_ids))
        for t in tree.turns
    ]


def tree_positions(tree: TreeSample) -> list[int]:
    """Per-token RoPE positions of a tree (what the training side derives from ``branches``)."""
    seg0 = len(tree.tokens) - sum(blen for blen, _ in tree.branches)
    positions = list(range(seg0))
    for blen, prefix in tree.branches:
        positions += range(prefix, prefix + blen)
    return positions
