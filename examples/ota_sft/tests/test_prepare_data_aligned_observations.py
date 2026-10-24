"""Tests for the observation-header rewrite of ``prepare_data_aligned.py``.

* synthetic cases for every branch of ``rewrite_observation`` / ``_rewrite_observations``;
* live-harness equivalence: Harbor's terminus-2 (``datagen_parity``) is run on the same assistant
  texts and terminal output, and must send exactly the observations we write. These tests need
  Harbor importable (e.g. a venv with harbor-marin installed) and skip otherwise;
* real data: after the rewrite, every main-loop observation is what the harness sends after the
  stored assistant text, except the unreproducible ones, and exactly the turns after those are
  unsupervised.

Run: python -m pytest examples/ota_sft/tests/test_prepare_data_aligned_observations.py -q
"""

import asyncio
import collections
import glob
import json
from pathlib import Path

import pytest

from examples.ota_sft.prepare_data_aligned import (
    CONFIRM_PREFIX,
    CONFIRM_SUFFIX,
    HARBOR_JSON_PARSER,
    NOT_OBSERVATION_PREFIXES,
    PARSE_ERROR_PREFIX,
    PARSE_ERROR_SUFFIX,
    WARNINGS_PREFIX,
    _rewrite_observations,
    convert_rows,
    harness_observation,
    is_main_loop_turn,
    harness_turn,
    load_harbor_json_parser,
    rewrite_observation,
    split_observation,
)

SNAPSHOT = Path("/data/murray/ota_sft/datasets/OpenThoughts-Agent-SFT-10K")
pytestmark = pytest.mark.skipif(not HARBOR_JSON_PARSER.exists(), reason="Harbor checkout not available")

TERM = "New Terminal Output:\n$ ls\nfile\n$ "
INITIAL = "You are an AI assistant...\nTask Description:\ndo it\n\nCurrent terminal state:\nCurrent Terminal Screen:\n$ \n"
SCREEN = "Current Terminal Screen:\n$ "
BEFORE = "- Extra text detected before JSON object"
AFTER = "- Extra text detected after JSON object"


@pytest.fixture(scope="module")
def parser():
    return load_harbor_json_parser()


def js(*keys, complete=None, durations=None):
    commands = [{"keystrokes": k, "duration": (durations or [0.1] * len(keys))[i]} for i, k in enumerate(keys)]
    body = {"analysis": "a", "plan": "p", "commands": commands}
    if complete is not None:
        body["task_complete"] = complete
    return json.dumps(body, indent=2)


def think(text, reasoning="I will run ls."):
    return f"<think>{reasoning}</think>{text}"


def fenced(text):
    return f"```json\n{text}\n```"


def warn_obs(header, terminal=TERM):
    return f"{WARNINGS_PREFIX}WARNINGS: {header}\n\n{terminal}"


def confirm_obs(terminal=TERM):
    return f"{CONFIRM_PREFIX}{terminal}{CONFIRM_SUFFIX}"


# ---------------------------------------------------------------- parsing of stored observations


@pytest.mark.parametrize(
    "content, expected",
    [
        (TERM, ("continue", "", TERM)),
        (SCREEN, ("continue", "", SCREEN)),
        ("Previous command:\nsleep 9\n\nThe previous command timed out", ("continue", "", None)),
        (warn_obs(BEFORE), ("continue", f"WARNINGS: {BEFORE}", TERM)),
        (warn_obs(f"{BEFORE}\n{AFTER}", SCREEN), ("continue", f"WARNINGS: {BEFORE}\n{AFTER}", SCREEN)),
        (f"{PARSE_ERROR_PREFIX}ERROR: x\nWARNINGS: y{PARSE_ERROR_SUFFIX}", ("error", "ERROR: x\nWARNINGS: y", None)),
        (confirm_obs(), ("complete", None, TERM)),
        ("Here are the answers the other agent provided.\n\nA", ("unknown", None, None)),
        ("obs", ("unknown", None, None)),
    ],
)
def test_split_observation(content, expected):
    kind, header, terminal = split_observation(content)
    assert (kind, header) == expected[:2]
    if expected[2] is not None:
        assert terminal == expected[2]


def test_split_cuts_at_the_first_terminal_capture():
    # the terminal output may itself contain an observation-like paragraph later on
    content = warn_obs(BEFORE, TERM + "\n\nNew Terminal Output:\nlater")
    assert split_observation(content) == ("continue", f"WARNINGS: {BEFORE}", TERM + "\n\nNew Terminal Output:\nlater")


# ---------------------------------------------------------------- rewrite branches


def test_clean_json_keeps_plain_observation(parser):
    assert rewrite_observation(parser, js("ls\n"), TERM, False) == (TERM, "ok", False)


def test_fence_added_after_generation_gets_its_warning(parser):
    # datagen parsed the unfenced text (no warning); the stored, fenced text warns before and after
    assistant = think(fenced(js("ls\n")))
    new, reason, pending = rewrite_observation(parser, assistant, TERM, False)
    assert reason == "ok" and not pending
    assert new == warn_obs(f"{BEFORE}\n{AFTER}")


def test_missing_after_warning_is_added(parser):
    assistant = think(js("ls\n")) + "\n```"
    new, reason, _ = rewrite_observation(parser, assistant, warn_obs(BEFORE), False)
    assert reason == "ok" and new == warn_obs(f"{BEFORE}\n{AFTER}")


def test_trailing_fence_only_gets_the_after_warning(parser):
    new, reason, _ = rewrite_observation(parser, js("ls\n") + "\n```", TERM, False)
    assert reason == "ok" and new == warn_obs(AFTER)


def test_header_no_candidate_explains_is_unrepairable(parser):
    # datagen saw extra text that no candidate of the stored text has: the stored text was
    # changed in a way we cannot undo, so what was executed cannot be verified
    got = rewrite_observation(parser, js("ls\n"), warn_obs(AFTER), False)
    assert got == (None, "no_datagen_parse_reproduces_header", False)


def test_newline_warning_comes_from_the_same_parser(parser):
    # "Command 1 should end with newline ..." depends on the keystrokes; it is part of the header
    assistant = js("ls", "pwd\n")
    turn = harness_turn(parser, assistant)
    assert "Command 1 should end with newline" in turn.feedback
    new, reason, _ = rewrite_observation(parser, assistant, TERM, False)
    assert reason == "no_datagen_parse_reproduces_header" and new is None
    stored = warn_obs(turn.feedback[len("WARNINGS: ") :])
    assert rewrite_observation(parser, assistant, stored, False) == (stored, "ok", False)


def test_parse_error_message_is_recomputed(parser):
    assistant = think("I am not sure what to do.")
    turn = harness_turn(parser, assistant)
    stored = f"{PARSE_ERROR_PREFIX}ERROR: No valid JSON found in response{PARSE_ERROR_SUFFIX}"
    new, reason, pending = rewrite_observation(parser, assistant, stored, True)
    assert reason == "ok" and pending is True  # a parse error leaves the completion state alone
    assert new == f"{PARSE_ERROR_PREFIX}{turn.feedback}{PARSE_ERROR_SUFFIX}" and new != stored


def test_completion_confirmation_kept_and_sets_pending(parser):
    assistant = think(fenced(js(complete=True)))
    assert rewrite_observation(parser, assistant, confirm_obs(), False) == (confirm_obs(), "ok", True)


def test_completion_after_pending_ends_the_run(parser):
    assistant = js(complete=True)
    assert rewrite_observation(parser, assistant, confirm_obs(), True) == (None, "run_would_have_ended", True)


@pytest.mark.parametrize(
    "assistant, stored, reason",
    [
        ('{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n"}]}', warn_obs(BEFORE), "kind_continue_to_error"),
        (js("ls\n"), f"{PARSE_ERROR_PREFIX}ERROR: x{PARSE_ERROR_SUFFIX}", "kind_error_to_continue"),
        (js("ls\n", complete=True), TERM, "kind_continue_to_complete"),
        (js("ls\n"), confirm_obs(), "kind_complete_to_continue"),
    ],
)
def test_kind_changes_are_unrepairable(parser, assistant, stored, reason):
    new, got, _ = rewrite_observation(parser, assistant, stored, False)
    assert (new, got) == (None, reason)


def test_different_executed_commands_are_unrepairable(parser):
    # the stored text's first JSON object is inside the think block; the datagen parse that
    # reproduces the stored header ran the commands after the think block
    assistant = think(js("echo final\n"), reasoning=f"draft: {js('rm -rf draft\n')}")
    stored_header = harness_turn(parser, js("echo final\n")).feedback
    stored = warn_obs(stored_header[len("WARNINGS: ") :]) if stored_header else TERM
    assert rewrite_observation(parser, assistant, stored, False) == (None, "executed_commands_differ", False)


def test_duration_cap_is_what_counts_as_executed(parser):
    # durations above 60 s execute as 60 s, so 90 vs 120 is the same execution
    a = think(js("make\n", durations=[90]))
    assert harness_turn(parser, a).commands == (("make\n", 60),)


# ---------------------------------------------------------------- row-level rule


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_unreproducible_unsupervises_every_later_turn_only(parser):
    good = think(fenced(js("ls\n")))
    msgs = _msgs(
        ("user", "sys"),
        ("assistant", good),
        ("user", TERM),
        ("assistant", js("ls\n", complete=True)),
        ("user", TERM),  # harness would have asked for confirmation: unrepairable
        ("assistant", good),
        ("user", TERM),
        ("assistant", good),
    )
    stats = collections.Counter()
    _rewrite_observations(parser, msgs, stats)
    assert [m.get("loss", True) for m in msgs if m["role"] == "assistant"] == [True, True, False, False]
    assert msgs[2]["content"] == warn_obs(f"{BEFORE}\n{AFTER}")
    assert msgs[4]["content"] == TERM and msgs[6]["content"] == TERM  # left as stored
    assert stats["unreproducible_kind_continue_to_complete"] == 1
    assert stats["turns_after_unreproducible_unsupervised"] == 2
    assert stats["observations_checked"] == 3 and stats["observations_rewritten"] == 1


def test_summary_and_handoff_prompts_are_not_observations(parser):
    answers_prompt, handoff = NOT_OBSERVATION_PREFIXES
    msgs = _msgs(
        ("user", "sys"),
        ("user", "You are picking up work from a previous AI agent on this task:"),
        ("assistant", "questions"),
        ("user", handoff + "A\n\nContinue working on this task"),
        ("assistant", js(complete=True)),
        ("user", confirm_obs()),
        ("assistant", js("ls\n")),
        ("user", TERM),
        ("user", "You are about to hand off your work to another AI agent."),
        ("assistant", "summary"),
        ("user", answers_prompt + "Q"),
        ("assistant", "answers"),
    )
    before = json.loads(json.dumps(msgs))
    stats = collections.Counter()
    _rewrite_observations(parser, msgs, stats)
    assert msgs == before and stats["observations_checked"] == 2
    assert not any("loss" in m for m in msgs)


def test_handoff_resets_completion_state(parser):
    _, handoff = NOT_OBSERVATION_PREFIXES
    msgs = _msgs(
        ("user", "sys"),
        ("assistant", js(complete=True)),
        ("user", confirm_obs()),
        ("assistant", "questions"),
        ("user", handoff + "A"),
        ("assistant", js(complete=True)),
        ("user", confirm_obs()),  # pending was reset by the handoff: a first confirmation again
        ("assistant", js("ls\n")),
    )
    stats = collections.Counter()
    _rewrite_observations(parser, msgs, stats)
    assert stats["unreproducible_observations"] == 0


def test_convert_rows_rewrites_after_warning_strip_and_before_dedup(parser):
    warn = "[SYSTEM WARNING] Context is 71% full (1/2 tokens). x"
    row = {
        "conversations": _msgs(("user", "sys"), ("assistant", think(fenced(js("ls\n")))), ("user", f"{warn}\n\n{TERM}"), ("assistant", js("pwd\n"))),
        "task": "t",
        "trace_source": "main",
    }
    out, stats = convert_rows([row, row], parser)
    assert out[0]["messages"][2]["content"] == warn_obs(f"{BEFORE}\n{AFTER}")
    assert out[0]["messages"][3]["loss"] is False  # read the stripped warning
    assert out[1]["messages"][1]["loss"] is False  # duplicate of row 0's first turn
    assert stats["observations_rewritten"] == 2


# ---------------------------------------------------------------- live harness equivalence


def _harness_prompts(assistant_texts: list[str], terminal_parts: list[str]) -> list[str]:
    """Run Harbor terminus-2 (datagen_parity) on scripted model texts and terminal outputs and
    return the observations it sends after each model text."""
    t2 = pytest.importorskip("harbor.agents.terminus_2.terminus_2")
    from harbor.agents.base import TurnCapExhaustedError
    from harbor.llms.base import LLMResponse
    from harbor.models.agent.context import AgentContext

    class LLM:
        def __init__(self):
            self.texts, self.prompts = list(assistant_texts) + [js("true\n")], []

        async def call(self, prompt, message_history=None, **kwargs):
            self.prompts.append(prompt)
            return LLMResponse(content=self.texts.pop(0))

        def get_model_context_limit(self):
            return 10**9

        def get_model_output_limit(self):
            return None

    class Session:
        def __init__(self):
            # initial screen, the scripted outputs, and one for the trailing filler model turn
            self.outputs = ["Current Terminal Screen:\n$ "] + list(terminal_parts) + ["$ "]

        async def is_session_alive(self):
            return True

        async def send_keys(self, *args, **kwargs):
            pass

        async def get_incremental_output(self):
            assert self.outputs, "harness consumed more terminal output than the data has"
            return self.outputs.pop(0)

        async def capture_pane(self, capture_entire=False):
            return "$ "

    llm = LLM()

    class Agent(t2.Terminus2):
        def _init_llm(self, **kwargs):
            return llm

    import tempfile

    agent = Agent(
        logs_dir=Path(tempfile.mkdtemp()),
        model_name="hosted_vllm/m",
        datagen_parity=True,
        max_turns=len(assistant_texts) + 1,
        model_info={"max_input_tokens": 10**9, "max_output_tokens": 10**6, "input_cost_per_token": 0, "output_cost_per_token": 0},
    )

    async def zero(*args, **kwargs):
        return 0

    agent._acount_total_tokens = zero
    agent._limit_output_length = lambda output, max_bytes=10000: output  # data keeps the limited part
    agent._session = Session()
    try:
        asyncio.run(agent.run("task", environment=object(), context=AgentContext()))
    except TurnCapExhaustedError:
        pass
    return llm.prompts[1 : len(assistant_texts) + 1]


SYNTHETIC_RUN = [
    (js("ls\n"), TERM),
    (think(fenced(js("ls\n"))), TERM),
    (think("no json here"), None),
    (js("ls", "pwd\n"), TERM),
    (think(js("cat x\n")) + "\n```", SCREEN),
    (js(complete=True), TERM),
    (js("ls\n"), TERM),
    (think(fenced(js(complete=True))), SCREEN),
    (js(complete=True), TERM),
]


def test_live_harness_matches_every_branch(parser):
    texts = [a for a, _ in SYNTHETIC_RUN]
    terminals = [t for _, t in SYNTHETIC_RUN if t is not None]
    sent = _harness_prompts(texts, terminals)
    pending, expected = False, []
    terminal_iter = iter(terminals)
    for text, _terminal in SYNTHETIC_RUN:
        turn = harness_turn(parser, text)
        obs = harness_observation(turn, next(terminal_iter) if turn.kind != "error" else None, pending)
        if turn.kind == "complete":
            pending = True
        elif turn.kind == "continue":
            pending = False
        if obs is None:
            break
        expected.append(obs)
    kinds = {harness_turn(parser, t).kind for t in texts}
    assert kinds == {"error", "complete", "continue"}
    assert sent == expected and len(expected) == len(SYNTHETIC_RUN) - 1  # the last completion ends the run


# ---------------------------------------------------------------- real data


@pytest.fixture(scope="module")
def real(parser):
    if not SNAPSHOT.exists():
        pytest.skip("OTA snapshot not available")
    import pyarrow.parquet as pq

    rows = pq.read_table(sorted(glob.glob(f"{SNAPSHOT}/**/*.parquet", recursive=True))).to_pylist()
    out, stats = convert_rows(rows, parser)
    return rows, out, stats


def _main_loop_segments(messages):
    """Runs of (assistant, observation) pairs of the harness main loop, with the index of the turn."""
    segment = []
    for i in range(1, len(messages)):
        m, prev = messages[i], messages[i - 1]
        if m["role"] != "user" or prev["role"] != "assistant":
            continue
        if m["content"].startswith(NOT_OBSERVATION_PREFIXES):
            if segment:
                yield segment
            segment = []
            continue
        segment.append((i - 1, prev["content"], m["content"]))
    if segment:
        yield segment


def test_real_every_observation_is_what_the_harness_sends(real, parser):
    rows, out, stats = real
    reproduced = unreproduced = after_unreproduced = 0
    for rec in out:
        msgs = rec["messages"]
        broken_at = None
        for segment in _main_loop_segments(msgs):
            pending = False
            for turn_index, assistant, observation in segment:
                kind, _, terminal = split_observation(observation)
                assert kind != "unknown"
                if broken_at is not None:
                    after_unreproduced += 1  # left as stored; its context is already unreproducible
                    continue
                turn = harness_turn(parser, assistant)
                expected = harness_observation(turn, terminal, pending) if turn.kind == kind else None
                # the text can match while the commands the datagen harness ran differed (that
                # needs the datagen-parse candidates, so it comes from rewrite_observation)
                _, reason, _ = rewrite_observation(parser, assistant, observation, pending)
                if expected == observation and reason != "executed_commands_differ":
                    reproduced += 1
                    pending = {"error": pending, "complete": True, "continue": False}[turn.kind]
                else:
                    unreproduced += 1
                    broken_at = turn_index
        # exactly the turns after the first unreproducible observation lost supervision for it
        if broken_at is not None:
            assert all(m.get("loss", True) is False for m in msgs[broken_at + 1 :] if m["role"] == "assistant")
    assert unreproduced == stats["unreproducible_observations"]
    assert reproduced + unreproduced + after_unreproduced == stats["observations_checked"]
    assert stats["user_after_assistant_not_observation"] == 0
    print(f"reproduced={reproduced} unreproducible={unreproduced} after={after_unreproduced} stats={dict(stats)}")


def test_real_live_harness_on_sampled_rows(real, parser):
    pytest.importorskip("harbor.agents.terminus_2.terminus_2")
    _, out, _ = real
    checked = 0
    for rec in out[::25]:
        for segment in _main_loop_segments(rec["messages"]):
            # stop at the first unreproducible observation (left as stored)
            ok_pairs, pending = [], False
            for _, assistant, observation in segment:
                turn = harness_turn(parser, assistant)
                kind, _, terminal = split_observation(observation)
                if turn.kind != kind or harness_observation(turn, terminal, pending) != observation:
                    break
                ok_pairs.append((assistant, observation))
                pending = {"error": pending, "complete": True, "continue": False}[turn.kind]
            if not ok_pairs:
                continue
            texts = [a for a, _ in ok_pairs]
            terminals = [split_observation(o)[2] for _, o in ok_pairs if split_observation(o)[0] != "error"]
            assert _harness_prompts(texts, terminals) == [o for _, o in ok_pairs]
            checked += len(ok_pairs)
    assert checked > 1000
    print(f"live-harness checked observations: {checked}")


# ---------------------------------------------------------------- parse failures are never trained


def test_main_loop_parse_failure_is_unsupervised(parser):
    _, handoff = NOT_OBSERVATION_PREFIXES
    bad = think("I will just answer in prose.")
    msgs = _msgs(
        ("user", INITIAL),
        ("assistant", bad),  # answers the initial prompt
        ("user", f"{PARSE_ERROR_PREFIX}{harness_turn(parser, bad).feedback}{PARSE_ERROR_SUFFIX}"),
        ("assistant", js("ls\n")),
        ("user", TERM),
        ("user", "You are about to hand off your work to another AI agent."),
        ("assistant", "a prose summary"),  # not a main-loop turn
    )
    row = {"conversations": msgs, "task": "t", "trace_source": "summarization-1-summary"}
    out, stats = convert_rows([row], parser)
    assert [m.get("loss", True) for m in out[0]["messages"] if m["role"] == "assistant"] == [False, True, True]
    assert stats["main_loop_targets_failing_parse"] == 1


def test_main_loop_turn_classification():
    _, handoff = NOT_OBSERVATION_PREFIXES
    msgs = _msgs(
        ("user", "sys"),
        ("user", "You are picking up work from a previous AI agent on this task:"),
        ("assistant", "questions"),
        ("user", handoff + "A"),
        ("assistant", "turn"),
        ("user", TERM),
        ("assistant", "turn"),
        ("user", "The next agent has a few questions for you, please answer each of them one by one in detail:\n\nQ"),
        ("assistant", "answers"),
    )
    assert [is_main_loop_turn(msgs, i) for i, m in enumerate(msgs) if m["role"] == "assistant"] == [False, True, True, False]
    assert is_main_loop_turn(_msgs(("user", INITIAL), ("assistant", "a")), 1)
    assert not is_main_loop_turn(_msgs(("user", "not a terminus prompt"), ("assistant", "a")), 1)


def test_real_no_supervised_main_loop_target_fails_parsing(real, parser):
    _, out, stats = real
    main_loop = failing_supervised = 0
    for rec in out:
        msgs = rec["messages"]
        for i, m in enumerate(msgs):
            if not is_main_loop_turn(msgs, i):
                continue
            main_loop += 1
            if m.get("loss", True) and harness_turn(parser, m["content"]).kind == "error":
                failing_supervised += 1
    assert failing_supervised == 0
    # every row's first turn answers a real terminus-2 initial prompt
    assert all(is_main_loop_turn(r["messages"], 1) for r in out if r["messages"][1]["role"] == "assistant")
    assert stats["main_loop_targets_failing_parse"] > 0
    print(f"main-loop turns={main_loop} failing parse={stats['main_loop_targets_failing_parse']}")
