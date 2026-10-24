"""SFT tree packing (``miles.rollout.sft_tree``) against the stock Qwen3 chat template.

Three layers:

* the think-stripping spec: a line-by-line python mirror of what the stock template renders for
  history, checked against the template itself on every edge case (so a template change fails here);
* per-turn prompts: ``build_turn_samples`` must equal ``apply_chat_template(..., tokenize=True)``;
* trees: every supervised token of a tree must be predicted from exactly the (token, position)
  context it has in its standalone per-turn sample, under the tree mask the trainer builds.
"""

import os
import random

import pytest
from transformers import AutoTokenizer

from miles.backends.training_utils.data.sft_tree_layout import build_tree_batch_layout, visible_keys
from miles.rollout.sft_tree import (
    assemble_tree,
    build_trees,
    build_turn_samples,
    expand_tree,
    tree_positions,
    validate_messages,
)

_TOKENIZER = os.environ.get("MILES_TEST_QWEN3_TOKENIZER", "/root/models/Qwen3-8B")
SUFFIX = "<|im_end|>\n"  # what examples/ota_sft supervises after the content


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(_TOKENIZER if os.path.isdir(_TOKENIZER) else "Qwen/Qwen3-8B")


# ---------------------------------------------------------------- the stripping spec


def spec_render(messages: list[dict], add_generation_prompt: bool) -> str:
    """The stock Qwen3 template for role/content-only messages without tools, transcribed to python."""
    out = ""
    if messages[0]["role"] == "system":
        out += "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
    last_query, multi_step_tool = len(messages) - 1, True
    for index in reversed(range(len(messages))):
        m = messages[index]
        c = m["content"]
        if multi_step_tool and m["role"] == "user" and not (
            c.startswith("<tool_response>") and c.endswith("</tool_response>")
        ):
            multi_step_tool, last_query = False, index
    for i, m in enumerate(messages):
        c = m["content"]
        if m["role"] == "user" or (m["role"] == "system" and i > 0):
            out += "<|im_start|>" + m["role"] + "\n" + c + "<|im_end|>\n"
        elif m["role"] == "assistant":
            reasoning = ""
            if "</think>" in c:
                reasoning = c.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                c = c.split("</think>")[-1].lstrip("\n")
            if i > last_query and (i == len(messages) - 1 or reasoning):
                out += "<|im_start|>assistant\n<think>\n" + reasoning.strip("\n") + "\n</think>\n\n" + c.lstrip("\n")
            else:
                out += "<|im_start|>assistant\n" + c
            out += "<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


def spec_history_content(content: str) -> str:
    """What an assistant turn before the last query renders as."""
    return content.split("</think>")[-1].lstrip("\n") if "</think>" in content else content


EDGE_CONTENTS = [
    "no think at all",
    "<think>\nreason\n</think>\n\nanswer",
    "<think>unclosed reasoning, so nothing is stripped",
    "reasoning without an opening tag</think>answer",
    "<think>a</think>b</think>c",
    "<think>\n\n</think>\n\nanswer after an empty think",
    "<think>\nonly a think\n</think>",
    "<think>\nthink then newlines\n</think>\n\n\n",
    "\n\nstarts with newlines, no think",
    "\n<think>\nr\n</think>\nanswer",
    "<think>r</think>\n\n\n  answer with spaces  \n",
    "",
    "   <think>r</think>leading spaces before the tag",
    "<think>a<think>b</think>nested open",
    "<think>r</think>\r\nanswer after CRLF",
    "<think>r</think>\t\nanswer after a tab",
    "</think>",
    "<think></think>",
    '<think>\nlet me run ls\n</think>\n\n{"analysis": "x", "commands": [{"keystrokes": "ls\\n"}]}',
]


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("content", EDGE_CONTENTS)
def test_template_matches_spec_on_history(tok, content, system):
    messages = [{"role": "user", "content": "first task"}, {"role": "assistant", "content": content}]
    messages += [{"role": "user", "content": "New Terminal Output:\n$ "}]
    if system:
        messages = [{"role": "system", "content": "be terse"}] + messages
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == spec_render(messages, add_generation_prompt=True)
    assert "<|im_start|>assistant\n" + spec_history_content(content) + "<|im_end|>\n" in rendered


@pytest.mark.parametrize("content", EDGE_CONTENTS)
def test_template_matches_spec_after_tool_responses(tok, content):
    """Assistant turns after the last real query (only <tool_response> users since) keep their reasoning."""
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "<tool_response>ok</tool_response>"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "<tool_response>ok</tool_response>"},
    ]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == spec_render(messages, add_generation_prompt=True)


def test_consecutive_users_and_handoff_shape(tok):
    messages = [
        {"role": "user", "content": "system prompt as a user turn"},
        {"role": "user", "content": "You are picking up work from a previous AI agent..."},
        {"role": "assistant", "content": "<think>q</think>questions"},
        {"role": "user", "content": "Here are the answers the other agent provided.\n\n<think>x</think>a"},
    ]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == spec_render(messages, add_generation_prompt=True)


# ---------------------------------------------------------------- per-turn prompts


def _reference_prompt_ids(tok, messages):
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return list(out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out)


def _conversation(contents: list[str], *, system: bool = False, loss: dict[int, bool] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}] if system else []
    for i, c in enumerate(contents):
        messages.append({"role": "user", "content": f"observation {i}\n$ "})
        a = {"role": "assistant", "content": c}
        if loss and i in loss:
            a["loss"] = loss[i]
        messages.append(a)
    return messages


def _assert_prompts_match_template(tok, messages):
    turns = build_turn_samples(tok, messages, target_suffix=SUFFIX)
    for t in turns:
        assert list(t.prompt_ids) == _reference_prompt_ids(tok, messages[: t.message_index])
        assert list(t.target_ids) == tok.encode(messages[t.message_index]["content"] + SUFFIX, add_special_tokens=False)
    return turns


@pytest.mark.parametrize("system", [False, True])
def test_turn_prompts_equal_apply_chat_template(tok, system):
    messages = _conversation(EDGE_CONTENTS, system=system)
    turns = _assert_prompts_match_template(tok, messages)
    assert [t.message_index for t in turns] == [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def test_loss_false_turn_stays_in_context_without_a_branch(tok):
    contents = ["<think>a</think>A", "<think>b</think>B", "<think>c</think>C"]
    messages = _conversation(contents, loss={1: False})
    turns = _assert_prompts_match_template(tok, messages)
    assert [t.message_index for t in turns] == [1, 5]
    # turn 1 (loss false) is still rendered, think-stripped, in the last prompt
    assert "<|im_start|>assistant\nB<|im_end|>" in tok.decode(turns[-1].prompt_ids)


def test_trailing_user_and_consecutive_users(tok):
    messages = [
        {"role": "user", "content": "u0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>x</think>a0"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "trailing user, no target"},
    ]
    turns = _assert_prompts_match_template(tok, messages)
    assert [t.message_index for t in turns] == [2, 4]


@pytest.mark.parametrize(
    "bad",
    [
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "r"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "tool_calls": []}],
        [{"role": "user", "content": "q"}, {"role": "tool", "content": "t"}],
        [{"role": "user", "content": ["parts"]}],
        [{"role": "user", "content": "q"}, {"role": "system", "content": "late system"}],
        [{"role": "user", "content": "q", "loss": False}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "loss": "no"}],
    ],
)
def test_unrenderable_input_is_rejected(bad):
    with pytest.raises(AssertionError):
        validate_messages(bad)


@pytest.mark.parametrize(
    "messages",
    [
        # after a <tool_response> user the template keeps earlier reasoning: not this packing's shape
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>r</think>a"},
            {"role": "user", "content": "<tool_response>x</tool_response>"},
            {"role": "assistant", "content": "b"},
        ],
        # a supervised turn right after another assistant turn (even an unsupervised one)
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "loss": False},
            {"role": "assistant", "content": "b"},
        ],
        [{"role": "assistant", "content": "a"}],
        [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
    ],
)
def test_supervised_turn_must_follow_a_user_query(tok, messages):
    with pytest.raises(AssertionError):
        build_turn_samples(tok, messages, target_suffix=SUFFIX)


def test_all_loss_false_trace_has_no_turns(tok):
    messages = _conversation(["a", "b"], loss={0: False, 1: False})
    turns = build_turn_samples(tok, messages, target_suffix=SUFFIX)
    assert turns == []
    trees, stats = build_trees(turns, max_context_len=32768, max_tree_tokens=32768)
    assert trees == [] and stats.num_turns == 0 and stats.tree_tokens == 0


# ---------------------------------------------------------------- trees


def check_tree(tree, *, rows=None):
    """Every supervised token: label and visible (token, position) context equal its standalone sample's."""
    layout = build_tree_batch_layout([len(tree.tokens)], [list(tree.branches)], pad=0)
    assert list(layout.positions) == tree_positions(tree)
    trunk_len = len(tree.tokens) - tree.response_length
    # where each turn's target tokens live in the tree
    spans, cursor = [], trunk_len
    for i, t in enumerate(tree.turns):
        cursor += 0 if i == 0 else 1  # the branch's leading duplicate token
        spans.append((cursor, t))
        cursor += len(t.target_ids)
    assert cursor == len(tree.tokens)
    mask = [0] * trunk_len + list(tree.loss_mask)
    supervised = {start + j for start, t in spans for j in range(len(t.target_ids))}
    assert {p for p, m in enumerate(mask) if m} == supervised
    # no supervised prediction crosses a segment boundary: label row p and predictor row p - 1 share a segment
    # (the inline target's first token is predicted by the trunk's last token, a branch's by its d_k)
    seg_of = [max(i for i, b in enumerate(layout.cu_seqlens[:-1]) if b <= r) for r in range(len(tree.tokens))]
    for p in supervised:
        assert seg_of[p - 1] == seg_of[p], f"label row {p} predicted across a segment boundary"
    expanded = expand_tree(tree)
    for (start, t), (std_tokens, std_mask) in zip(spans, expanded, strict=True):
        n_prompt = len(t.prompt_ids)
        assert std_mask == [0] * n_prompt + [1] * len(t.target_ids)
        for j in range(len(t.target_ids)) if rows is None else rows(len(t.target_ids)):
            p = start + j
            assert tree.tokens[p] == std_tokens[n_prompt + j]
            # the row predicting p is p - 1 in the flat layout (what the loss code uses)
            ctx_tree = [(tree.tokens[i], layout.positions[i]) for i in visible_keys(layout, p - 1)]
            ctx_std = [(std_tokens[i], i) for i in range(n_prompt + j)]
            assert ctx_tree == ctx_std, f"turn {t.message_index} token {j}"


def _trees(tok, messages, max_context_len=32768, max_tree_tokens=32768):
    turns = build_turn_samples(tok, messages, target_suffix=SUFFIX)
    return build_trees(turns, max_context_len=max_context_len, max_tree_tokens=max_tree_tokens)


def test_one_turn_tree_is_the_standalone_sample(tok):
    (tree,), stats = _trees(tok, _conversation(["<think>r</think>answer"]))
    ((std_tokens, std_mask),) = expand_tree(tree)
    assert list(tree.tokens) == std_tokens and tree.branches == ()
    assert [0] * (len(std_tokens) - tree.response_length) + list(tree.loss_mask) == std_mask
    check_tree(tree)
    assert stats.tree_tokens == stats.expanded_tokens


def test_many_turns_one_tree(tok):
    contents = [f"<think>\nthinking {i} " + "x " * (i * 7) + f"\n</think>\n\nanswer {i}" for i in range(9)]
    (tree,), stats = _trees(tok, _conversation(contents))
    assert len(tree.branches) == 8 and stats.num_prefix_breaks == 0
    check_tree(tree)
    assert stats.tree_tokens < stats.expanded_tokens
    assert stats.target_tokens == sum(tree.loss_mask)


def test_mixed_edge_contents(tok):
    """Edge contents (unclosed think, empty content, no think, ...) all train exactly; prefix breaks split trees."""
    trees, stats = _trees(tok, _conversation(EDGE_CONTENTS, system=True))
    assert sum(len(t.turns) for t in trees) == len(EDGE_CONTENTS)
    for tree in trees:
        check_tree(tree)
    assert stats.num_trees == stats.num_prefix_breaks + 1


def test_newline_merge_breaks_the_prefix(tok):
    """A turn rendered as '...assistant\\n\\n\\n...' merges newlines across the generation prompt."""
    turns = build_turn_samples(tok, _conversation(["\n\nno think, leading newlines", "b"]), target_suffix=SUFFIX)
    assert turns[1].prompt_ids[: len(turns[0].prompt_ids)] != turns[0].prompt_ids
    trees, stats = build_trees(turns, max_context_len=32768, max_tree_tokens=32768)
    assert stats.num_prefix_breaks == 1 and len(trees) == 2
    for tree in trees:
        check_tree(tree)


def test_budget_splits_into_subtrees(tok):
    contents = [f"<think>{'t ' * 50}</think>a{i}" for i in range(6)]
    turns = build_turn_samples(tok, _conversation(contents), target_suffix=SUFFIX)
    whole = assemble_tree(turns)
    budget = len(whole.tokens) // 2
    trees, stats = build_trees(turns, max_context_len=budget, max_tree_tokens=budget)
    assert len(trees) >= 2 and all(len(t.tokens) <= budget for t in trees)
    assert [t.message_index for tree in trees for t in (*tree.turns[1:], tree.turns[0])] == [
        t.message_index for t in turns
    ]
    for tree in trees:
        check_tree(tree)
    # the split is optimal: no other consecutive split under the budget uses fewer tokens
    assert stats.tree_tokens == _brute_force_min_tokens(turns, budget)


def _brute_force_min_tokens(turns, budget):
    n, best = len(turns), None
    for cuts in range(1 << (n - 1)):
        groups, start = [], 0
        for i in range(1, n):
            if cuts >> (i - 1) & 1:
                groups.append(turns[start:i])
                start = i
        groups.append(turns[start:])
        sizes = [len(assemble_tree(g).tokens) for g in groups]
        if max(sizes) <= budget:
            best = sum(sizes) if best is None else min(best, sum(sizes))
    return best


def test_overlong_turns_are_dropped_whole(tok):
    contents = ["<think>short</think>a", "<think>" + "long " * 400 + "</think>b", "<think>s</think>c"]
    turns = build_turn_samples(tok, _conversation(contents), target_suffix=SUFFIX)
    cutoff = turns[1].length - 1
    trees, stats = build_trees(turns, max_context_len=cutoff, max_tree_tokens=cutoff)
    kept = {t.message_index for tree in trees for t in tree.turns}
    assert turns[1].message_index not in kept and stats.num_dropped_turns >= 1
    assert stats.dropped_target_tokens >= len(turns[1].target_ids)
    for tree in trees:
        assert all(t.length <= cutoff for t in tree.turns)
        check_tree(tree)


def test_budget_boundary_is_inclusive(tok):
    turns = build_turn_samples(tok, _conversation(["<think>a</think>A", "<think>b</think>B"]), target_suffix=SUFFIX)
    size = len(assemble_tree(turns).tokens)
    assert len(build_trees(turns, max_context_len=size, max_tree_tokens=size)[0]) == 1
    assert len(build_trees(turns, max_context_len=size - 1, max_tree_tokens=size - 1)[0]) == 2


def test_multiple_trees_in_one_micro_batch(tok):
    """Packing several trees back to back keeps each one's mask and positions to itself."""
    rng = random.Random(0)
    trees = []
    for n in (1, 3, 5):
        contents = [f"<think>{'w ' * rng.randint(1, 30)}</think>{i}" for i in range(n)]
        trees += _trees(tok, _conversation(contents))[0]
    lengths = [len(t.tokens) for t in trees]
    layout = build_tree_batch_layout(lengths, [list(t.branches) for t in trees], pad=5)
    offsets = [sum(lengths[:i]) for i in range(len(trees))]
    for tree, off in zip(trees, offsets, strict=True):
        single = build_tree_batch_layout([len(tree.tokens)], [list(tree.branches)], pad=0)
        assert layout.positions[off : off + len(tree.tokens)] == single.positions
        for row in range(len(tree.tokens)):
            assert [k - off for k in visible_keys(layout, off + row)] == visible_keys(single, row)
    assert layout.positions[-5:] == tuple(range(5)) and layout.cu_seqlens[-1] == sum(lengths) + 5
