"""Tests for ``prepare_data_aligned.py``: synthetic edge cases plus checks on the real OTA snapshot.

Run: python -m pytest examples/ota_sft/tests/test_prepare_data_aligned.py -q
(the real-data tests skip when the snapshot or the Harbor checkout is not on this machine).
"""

import glob
from pathlib import Path

import pytest

from examples.ota_sft.prepare_data_aligned import (
    NEW_NEWLINE_RULE,
    OLD_NEWLINE_RULE,
    convert_rows,
    normalize_system_prompt,
    split_observation,
    strip_system_warnings,
)

SNAPSHOT = Path("/data/murray/ota_sft/datasets/OpenThoughts-Agent-SFT-10K")
HARBOR_TEMPLATE = Path("/data/murray/ota_sft/src/harbor-marin/src/harbor/agents/terminus_2/templates/terminus-json-plain.txt")
TIME_W = (
    "[SYSTEM WARNING] You have approximately 5 minutes remaining before the task times out. "
    "Please wrap up your work and finalize your solution as soon as possible."
)
CTX_W = (
    "[SYSTEM WARNING] Context is 71% full (23,207/32,768 tokens). Your conversation will be summarized soon "
    "to free up space. Make sure your current approach and key findings are clearly stated so they survive "
    "summarization."
)


def _row(*messages, task="t", source="main"):
    return {"conversations": [{"role": r, "content": c} for r, c in messages], "task": task, "trace_source": source}


# ---------------------------------------------------------------- system prompt normalization


def test_normalize_replaces_old_rule_line():
    old = f"intro\n{OLD_NEWLINE_RULE}\n- For special key sequences"
    assert normalize_system_prompt(old) == f"intro\n{NEW_NEWLINE_RULE}\n- For special key sequences"


def test_normalize_is_noop_and_idempotent():
    new = f"intro\n{NEW_NEWLINE_RULE}\nrest"
    assert normalize_system_prompt(new) == new
    assert normalize_system_prompt(normalize_system_prompt(f"a\n{OLD_NEWLINE_RULE}")) == f"a\n{NEW_NEWLINE_RULE}"


def test_normalize_rejects_double_occurrence():
    with pytest.raises(AssertionError):
        normalize_system_prompt(f"{OLD_NEWLINE_RULE}\n{OLD_NEWLINE_RULE}")


def test_normalize_rejects_old_rule_inside_a_line():
    # e.g. quoted by the terminal output: we must not silently leave or half-edit it
    with pytest.raises(AssertionError):
        normalize_system_prompt(f"$ echo '{OLD_NEWLINE_RULE}'")


def test_normalize_rule_at_first_and_last_line():
    assert normalize_system_prompt(OLD_NEWLINE_RULE) == NEW_NEWLINE_RULE
    assert normalize_system_prompt(f"x\n{OLD_NEWLINE_RULE}") == f"x\n{NEW_NEWLINE_RULE}"


# ---------------------------------------------------------------- warning stripping


@pytest.mark.parametrize(
    "content, expected, n",
    [
        ("New Terminal Output:\n$ ls", "New Terminal Output:\n$ ls", 0),
        (f"{TIME_W}\n\nNew Terminal Output:\n$ ls", "New Terminal Output:\n$ ls", 1),
        (f"{CTX_W}\n\n{TIME_W}\n\nPrevious response had warnings:\nWARNINGS: - x", "Previous response had warnings:\nWARNINGS: - x", 2),
        (f"{TIME_W}\n\n", "", 1),
        (f"{TIME_W}\n\n\nX", "\nX", 1),  # only the separating blank line belongs to the warning
        ("", "", 0),
    ],
)
def test_strip_warnings(content, expected, n):
    assert strip_system_warnings(content) == (expected, n)


@pytest.mark.parametrize(
    "content",
    [
        f"New Terminal Output:\n{TIME_W}\n\n",  # not at the start
        f"{TIME_W}\nNew Terminal Output:",  # no blank line after it
        f"{TIME_W}",  # no separator at all
        f"{TIME_W}\n\nx\n{CTX_W}\n\n",  # a second warning later in the message
        f" {TIME_W}\n\nx",  # leading space
    ],
)
def test_strip_warnings_rejects_unexpected_shapes(content):
    with pytest.raises(AssertionError):
        strip_system_warnings(content)


# ---------------------------------------------------------------- row conversion


def test_turn_after_warning_is_unsupervised_and_only_that_turn():
    out, stats = convert_rows(
        [_row(("user", "sys"), ("assistant", "a1"), ("user", f"{TIME_W}\n\nobs"), ("assistant", "a2"), ("user", "o"), ("assistant", "a3"))]
    )
    msgs = out[0]["messages"]
    assert [m.get("loss", True) for m in msgs if m["role"] == "assistant"] == [True, False, True]
    assert msgs[2]["content"] == "obs"
    assert stats["warnings_stripped"] == 1 and stats["turns_after_warning_unsupervised"] == 1


def test_warning_in_any_user_message_since_last_turn_unsupervises_it():
    # user(warned) then user(clean) then assistant: the assistant still read the warned message
    out, _ = convert_rows([_row(("user", "sys"), ("user", f"{TIME_W}\n\nobs"), ("user", "p"), ("assistant", "a"), ("user", "o"), ("assistant", "b"))])
    assert [m.get("loss", True) for m in out[0]["messages"] if m["role"] == "assistant"] == [False, True]


def test_non_alternating_rows_are_kept_verbatim():
    rows = [_row(("user", "sys"), ("user", "You are picking up work"), ("assistant", "<think>q</think>Q"), ("user", "h"), ("assistant", "a"))]
    out, stats = convert_rows(rows)
    assert [(m["role"], m["content"]) for m in out[0]["messages"]] == [
        (c["role"], c["content"]) for c in rows[0]["conversations"]
    ]
    assert stats["non_alternating_rows"] == 1


def test_duplicate_turns_supervised_once():
    hist = [("user", "sys"), ("assistant", "<think>r1</think>a1"), ("user", "o1")]
    rows = [
        _row(*hist, ("user", "summarize"), ("assistant", "S"), source="summarization-1-summary"),
        _row(*hist, ("user", "summarize"), ("assistant", "S"), ("user", "questions"), ("assistant", "A"), source="summarization-1-answers"),
    ]
    out, stats = convert_rows(rows)
    sup = [[m.get("loss", True) for m in r["messages"] if m["role"] == "assistant"] for r in out]
    assert sup == [[True, True], [False, False, True]]
    assert stats["duplicate_turns_unsupervised"] == 2


def test_duplicate_detection_uses_the_rendered_context():
    # identical after think stripping -> same rendered prompt -> same sample; different target -> distinct
    a = _row(("user", "u"), ("assistant", "<think>x</think>A"), ("user", "o"), ("assistant", "T"))
    b = _row(("user", "u"), ("assistant", "<think>y</think>\nA"), ("user", "o"), ("assistant", "T"))
    c = _row(("user", "u"), ("assistant", "<think>y</think>A"), ("user", "o"), ("assistant", "T2"))
    out, _ = convert_rows([a, b, c])
    assert out[1]["messages"][3]["loss"] is False  # same stripped history ("A" after lstrip) and target
    assert out[1]["messages"][1].get("loss", True) is True  # turn 1: same context "u" but a different target
    assert out[2]["messages"][3].get("loss", True) is True


def test_unclosed_think_history_is_not_stripped_for_dedup():
    a = _row(("user", "u"), ("assistant", "<think>x"), ("user", "o"), ("assistant", "T"))
    b = _row(("user", "u"), ("assistant", "<think>y"), ("user", "o"), ("assistant", "T"))
    out, _ = convert_rows([a, b])
    assert out[1]["messages"][3].get("loss", True) is True  # contexts differ: unclosed think stays verbatim


def test_order_and_metadata_preserved():
    rows = [_row(("user", str(i)), ("assistant", "a"), task=f"t{i}") for i in range(5)]
    out, _ = convert_rows(rows)
    assert [r["metadata"]["row_index"] for r in out] == list(range(5))
    assert [r["metadata"]["task"] for r in out] == [f"t{i}" for i in range(5)]


def test_rejects_unknown_role():
    with pytest.raises(AssertionError):
        convert_rows([_row(("user", "u"), ("tool", "x"), ("assistant", "a"))])


# ---------------------------------------------------------------- real snapshot


@pytest.fixture(scope="module")
def real():
    if not SNAPSHOT.exists():
        pytest.skip("OTA snapshot not available")
    import pyarrow.parquet as pq

    rows = pq.read_table(sorted(glob.glob(f"{SNAPSHOT}/**/*.parquet", recursive=True))).to_pylist()
    out, stats = convert_rows(rows)
    return rows, out, stats


def test_real_keeps_every_row_in_order(real):
    rows, out, stats = real
    assert len(out) == len(rows) == stats["rows"] == 10000
    assert [r["metadata"]["row_index"] for r in out] == list(range(10000))


def test_real_no_warning_left(real):
    _, out, _ = real
    assert not any("[SYSTEM WARNING]" in m["content"] for r in out for m in r["messages"] if m["role"] == "user")


def test_real_system_prompt_equals_harbor_template(real):
    if not HARBOR_TEMPLATE.exists():
        pytest.skip("Harbor checkout not available")
    header = HARBOR_TEMPLATE.read_text().format(instruction="\0", terminal_state="\0").split("Task Description:")[0]
    _, out, stats = real
    firsts = [r["messages"][0]["content"] for r in out]
    assert all(f.startswith(header) for f in firsts)
    assert stats["system_prompt_normalized"] == 559


def test_real_only_user_messages_changed(real):
    """Assistant texts are untouched; a user message changes only by the warning strip, the system
    prompt line, or a recomputed observation header (same kind, same terminal part)."""
    rows, out, _ = real
    for row, rec in zip(rows, out, strict=True):
        src = row["conversations"]
        assert len(src) == len(rec["messages"])
        for a, b in zip(src, rec["messages"], strict=True):
            assert a["role"] == b["role"]
            if a["role"] == "assistant":
                assert a["content"] == b["content"]
                continue
            stripped, _ = strip_system_warnings(normalize_system_prompt(a["content"]))
            if b["content"] == stripped:
                continue
            old_kind, _, old_terminal = split_observation(stripped)
            new_kind, _, new_terminal = split_observation(b["content"])
            assert old_kind == new_kind != "unknown" and old_terminal == new_terminal


def test_real_unsupervised_turns_are_exactly_the_warned_and_duplicate_ones(real):
    rows, out, stats = real
    warned = 0
    for row, rec in zip(rows, out, strict=True):
        prev_warned = False
        for a, b in zip(row["conversations"], rec["messages"], strict=True):
            if a["role"] == "user":
                prev_warned = prev_warned or "[SYSTEM WARNING]" in a["content"]
            else:
                if prev_warned:
                    warned += 1
                    assert b["loss"] is False
                prev_warned = False
    assert warned == stats["turns_after_warning_unsupervised"]
    unsup = sum(m.get("loss", True) is False for r in out for m in r["messages"] if m["role"] == "assistant")
    assert unsup == (
        stats["turns_after_warning_unsupervised"]
        + stats["duplicate_turns_unsupervised"]
        + stats["turns_after_unreproducible_unsupervised"]
        + stats["main_loop_targets_failing_parse_unsupervised"]
    )


def test_real_stats_snapshot(real):
    _, _, stats = real
    print(dict(stats))
    assert stats["non_alternating_rows"] == 817
    assert stats["trailing_user_dropped"] == 0
