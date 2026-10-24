"""SFT tree packing on real OTA traces (needs the local data; not run in CI).

* every supervised turn's prompt == ``apply_chat_template(messages[:i], add_generation_prompt=True,
  tokenize=True)`` with the served template, for every row of the aligned sample (every row shape:
  main / post-handoff / summary / answers, with and without ``loss: false``) and the first rows of the
  LF-parity jsonl;
* every tree passes ``check_tree`` (labels, contexts and positions equal the standalone samples', no
  prediction across a segment) at the first, second, middle and last token of every target.

Full-dataset verification (all 122389 turns of the aligned file) is ``examples/ota_sft/tree_stats.py
--verify-every 1``.

    python -m pytest tests/manual/sft_tree/test_sft_tree_real_traces.py
"""

import json
import os

import pytest
from transformers import AutoTokenizer

from miles.rollout.sft_tree import build_trees, build_turn_samples
from tests.fast.rollout.test_sft_tree import check_tree

TOKENIZER = os.environ.get("QWEN3_8B", "/root/models/Qwen3-8B")
TEMPLATE = os.environ.get("CHAT_TEMPLATE", "/root/dev/chat_template_step664.jinja")
ALIGNED_SAMPLE = os.environ.get("TREE_SAMPLE", "/root/dev/aligned_sample.jsonl")
LF_PARITY = os.environ.get("OTA_JSONL", "/root/datasets/ota_sft_10k.jsonl")
SUFFIX = "<|im_end|>\n"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def template(tok):
    with open(TEMPLATE) as f:
        template = f.read()
    assert template == tok.chat_template, "the served template must be the checkpoint tokenizer's"
    return template


def _rows(path, limit=None):
    with open(path) as f:
        rows = [json.loads(line) for _, line in zip(range(limit or 10**9), f, strict=False)]
    return rows


def _check_row(tok, template, messages):
    turns = build_turn_samples(tok, messages, target_suffix=SUFFIX, chat_template=template)
    for t in turns:
        history = [{"role": m["role"], "content": m["content"]} for m in messages[: t.message_index]]
        ref = tok.apply_chat_template(history, tokenize=True, add_generation_prompt=True, chat_template=template)
        ref = list(ref["input_ids"] if hasattr(ref, "keys") else ref)
        assert list(t.prompt_ids) == ref, f"turn {t.message_index}"
    supervised = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("loss", True)]
    assert [t.message_index for t in turns] == supervised
    trees, stats = build_trees(turns, max_context_len=32768, max_tree_tokens=32768)
    assert stats.num_prefix_breaks == 0
    for tree in trees:
        assert len(tree.tokens) <= 32768
        check_tree(tree, rows=lambda n: sorted({0, min(1, n - 1), n // 2, n - 1}))
    return trees, stats


@pytest.mark.parametrize("index", range(32))
def test_aligned_sample_row(tok, template, index):
    rows = _rows(ALIGNED_SAMPLE)
    _check_row(tok, template, rows[index]["messages"])


@pytest.mark.parametrize("index", range(0, 200, 10))
def test_lf_parity_row(tok, template, index):
    rows = _rows(LF_PARITY, limit=200)
    _check_row(tok, template, rows[index]["messages"])
