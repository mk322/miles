"""examples/ota_sft rollout: the tree path's sample grouping, and the verbatim path left untouched."""

import os
from argparse import Namespace

import pytest
from transformers import AutoTokenizer

import examples.ota_sft.ota_sft_rollout as rollout
from miles.rollout.sft_tree import build_trees, build_turn_samples, expand_tree
from miles.utils.types import Sample

_TOKENIZER = os.environ.get("MILES_TEST_QWEN3_TOKENIZER", "/root/models/Qwen3-8B")


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(_TOKENIZER if os.path.isdir(_TOKENIZER) else "Qwen/Qwen3-8B")


@pytest.fixture
def template_file(tok, tmp_path):
    path = tmp_path / "chat_template.jinja"
    path.write_text(tok.chat_template)
    return str(path)


class _Buffer:
    def __init__(self, conversations):
        self.samples = [[Sample(index=i, group_index=i, prompt=c)] for i, c in enumerate(conversations)]

    def get_samples(self, n):
        assert n == len(self.samples)
        return self.samples


def _args(template_file, tree: bool, **kw):
    return Namespace(
        rollout_max_context_len=32768,
        hf_checkpoint=_TOKENIZER,
        rollout_batch_size=4,
        sft_tree_attention=tree,
        sft_tree_chat_template=template_file,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=kw.get("max_tokens_per_gpu", 32768),
    )


def _conversations():
    def conv(contents, loss=None):
        messages = []
        for i, c in enumerate(contents):
            messages.append({"role": "user", "content": f"obs {i}"})
            a = {"role": "assistant", "content": c}
            if loss is not None:
                a["loss"] = loss[i]
            messages.append(a)
        return messages

    return [
        conv(["<think>a</think>A", "<think>b</think>B", "<think>c</think>C"]),
        conv(["<think>x</think>X"]),
        conv(["<think>p</think>P", "<think>q</think>Q"], loss=[False, False]),  # no trainable turn
        [{"role": "user", "content": "u0"}, {"role": "user", "content": "u1"}] + conv(["<think>y</think>Y", "Z"]),
    ]


@pytest.fixture(autouse=True)
def _fresh_module_state(tok, monkeypatch):
    monkeypatch.setattr(rollout, "_TOKENIZER", tok)
    monkeypatch.setattr(rollout, "_CHAT_TEMPLATE", None)


def test_tree_rollout_groups_trees_by_trace(tok, template_file):
    convs = _conversations()
    out = rollout.generate_rollout(_args(template_file, True), 0, _Buffer(convs))
    assert len(out) == len(convs)
    for i, ((samples), conv) in enumerate(zip(out, convs, strict=True)):
        (siblings,) = samples
        assert all(s.rollout_id == i and s.index == i for s in siblings)
        turns = build_turn_samples(tok, conv, target_suffix=rollout.TARGET_SUFFIX, chat_template=tok.chat_template)
        trees, _ = build_trees(turns, max_context_len=32768, max_tree_tokens=32768)
        if not trees:
            (placeholder,) = siblings
            assert placeholder.loss_mask == [0] and placeholder.tree_branches is None
            continue
        assert [s.tokens for s in siblings] == [list(t.tokens) for t in trees]
        assert [s.tree_branches for s in siblings] == [[list(b) for b in t.branches] for t in trees]
        for s in siblings:
            s.validate()


def test_expanded_rollout_is_the_trees_standalone_samples(tok, template_file):
    convs = _conversations()
    trees_out = rollout.generate_rollout(_args(template_file, True), 0, _Buffer(convs))
    rollout._CHAT_TEMPLATE = None
    exp_out = rollout.generate_rollout_expanded(_args(template_file, True), 0, _Buffer(convs))
    for (tree_siblings,), (exp_siblings,), conv in zip(trees_out, exp_out, convs, strict=True):
        n_supervised = sum(sum(s.loss_mask) for s in tree_siblings)
        assert n_supervised == sum(sum(s.loss_mask) for s in exp_siblings)
        assert all(s.tree_branches is None for s in exp_siblings)
        assert len({s.rollout_id for s in exp_siblings}) == 1


def test_template_must_match_the_checkpoint(tok, tmp_path):
    bad = tmp_path / "other.jinja"
    bad.write_text(tok.chat_template + " ")
    with pytest.raises(AssertionError):
        rollout.generate_rollout(_args(str(bad), True), 0, _Buffer(_conversations()))


def test_split_respects_max_tokens_per_gpu(tok, template_file):
    convs = [[m for i in range(8) for m in ({"role": "user", "content": f"o{i} " * 50}, {"role": "assistant", "content": "<think>" + "t " * 60 + "</think>a"})]]
    args = _args(template_file, True, max_tokens_per_gpu=600)
    args.rollout_max_context_len = 600
    args.rollout_batch_size = 1
    ((siblings,),) = rollout.generate_rollout(args, 0, _Buffer(convs))
    assert len(siblings) > 1 and all(len(s.tokens) <= 600 for s in siblings)


def test_verbatim_path_is_unchanged(tok, template_file):
    """Flag off: the samples are exactly encode_conversation's (the LlamaFactory-parity path)."""
    convs = [c for c in _conversations() if [m["role"] for m in c] == ["user", "assistant"] * (len(c) // 2)]
    convs = [[{"role": m["role"], "content": m["content"]} for m in c] for c in convs][:2]
    args = _args(template_file, False)
    args.rollout_batch_size = len(convs)
    out = rollout.generate_rollout(args, 0, _Buffer(convs))
    for (sample,), conv in zip(out, convs, strict=True):
        ids, mask = rollout.encode_conversation(tok, conv, 32768)
        assert sample.tokens == ids and sample.tree_branches is None
        assert sample.loss_mask == mask[-sample.response_length :]
        # the verbatim slots are the LlamaFactory qwen3_nothink ones, byte for byte
        text = "".join(
            f"<|im_start|>user\n{u['content']}<|im_end|>\n<|im_start|>assistant\n{a['content']}<|im_end|>\n"
            for u, a in zip(conv[0::2], conv[1::2], strict=True)
        )
        assert tok.decode(ids) == text
    _ = expand_tree  # imported for parity with the tree tests
