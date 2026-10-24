"""Convert the OpenThoughts-Agent SFT snapshot into inference-aligned jsonl for tree-attention SFT.

Unlike ``prepare_data.py`` (LlamaFactory parity, 9,183 strictly alternating rows), this keeps every
row and edits the user messages so that each one is byte-identical to what our evaluation harness
(Harbor terminus-2 with ``datagen_parity``) sends:

* every row is kept, including the 817 summarization rows LlamaFactory drops for not alternating
  (summary / answers sub-calls and post-handoff main traces): the tree builder renders any role
  order exactly like the chat template;
* the 2025-12 rows carry an older line of the terminus-2 system prompt; it is replaced by the
  current line, which is the only difference to the prompt the evaluation harness sends;
* ``[SYSTEM WARNING]`` paragraphs (time left / context fill), which only the datagen harness
  emitted, are removed from the start of the observations. The assistant turn answering such an
  observation stays in the context but is not supervised (``"loss": false``): it was written after
  reading a warning the model will never see;
* the parser feedback that opens each main-loop observation ("Previous response had warnings:" /
  "Previous response had parsing errors:" / none) is recomputed from the stored assistant text with
  Harbor's own ``TerminusJSONPlainParser``, using the parity harness's observation construction
  (see ``rewrite_observation``). The datagen harness parsed a different text in about 17% of the
  turns: the stored text was reformatted after generation, e.g. by adding the trailing code fence.
  Our model emits the stored form, so at evaluation the harness reports on that form. The terminal
  output that follows the header is kept;
* rule: an assistant turn is supervised only if its full context is reproducible by the parity
  harness. When the recomputed parse would have changed what the harness executed (a parse error
  instead of executed commands or the reverse, a different completion step, different commands, or
  no plausible datagen parse reproduces the stored header), the terminal output after that turn
  cannot be repaired. The observation is then left as stored, and every later assistant turn of
  the row gets ``"loss": false``;
* a main-loop assistant turn (one that answers the initial prompt, an observation or a handoff) whose
  stored text fails ``parse_response`` (the harness would answer it with "Previous response had
  parsing errors") is not supervised: the model must not learn to emit it;
* an assistant turn whose context and target already occurred earlier in the dataset (summary and
  answers rows of one summarization share their history) is kept as context but not supervised
  again, so no turn is trained twice per epoch.

Row order is preserved (OpenThoughts-Agent trains with ``disable_shuffling=True``).

Usage:
  python examples/ota_sft/prepare_data_aligned.py \
      --src /data/murray/ota_sft/datasets/OpenThoughts-Agent-SFT-10K \
      --dst /data/murray/ota_sft/datasets/ota_sft_10k_aligned.jsonl
"""

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

OLD_NEWLINE_RULE = "- Most bash commands should end with a newline (\\n) to cause them to execute"
NEW_NEWLINE_RULE = "- You must end every command with a newline (\\n) or it will not execute."

# One or more warning paragraphs at the very start of an observation, each followed by a blank line.
_WARNINGS_AT_START = re.compile(r"\A(?:\[SYSTEM WARNING\][^\n]*\n\n)+")
_WARNING = "[SYSTEM WARNING]"


def normalize_system_prompt(content: str) -> str:
    """Replace the 2025-12 terminus-2 newline rule by the current one (a whole line, at most once)."""
    lines = content.split("\n")
    hits = [i for i, line in enumerate(lines) if line == OLD_NEWLINE_RULE]
    assert len(hits) <= 1, f"old newline rule occurs {len(hits)} times"
    for i in hits:
        lines[i] = NEW_NEWLINE_RULE
    out = "\n".join(lines)
    assert OLD_NEWLINE_RULE not in out, "old newline rule left inside a line"
    return out


def strip_system_warnings(content: str) -> tuple[str, int]:
    """Remove the leading ``[SYSTEM WARNING]`` paragraphs; returns the content and how many were removed."""
    m = _WARNINGS_AT_START.match(content)
    if m is None:
        assert _WARNING not in content, f"[SYSTEM WARNING] not at the start of the observation: {content[:200]!r}"
        return content, 0
    rest = content[m.end() :]
    assert _WARNING not in rest, f"[SYSTEM WARNING] left after the leading block: {rest[:200]!r}"
    return rest, m.group(0).count(_WARNING)


# ---------------------------------------------------------------- observation headers
#
# Harbor's terminus-2 (datagen_parity, JSON parser) builds the observation after a model turn in
# ``Terminus2._handle_llm_interaction`` / ``_run_agent_loop``; the strings below are that code's
# literals (checked against a live harness run by the tests).

HARBOR_JSON_PARSER = Path(
    os.environ.get(
        "HARBOR_JSON_PARSER",
        "/data/murray/ota_sft/src/harbor-marin/src/harbor/agents/terminus_2/terminus_json_plain_parser.py",
    )
)
PARSE_ERROR_PREFIX = "Previous response had parsing errors:\n"
PARSE_ERROR_SUFFIX = "\n\nPlease fix these issues and provide a proper JSON response."
WARNINGS_PREFIX = "Previous response had warnings:\n"
CONFIRM_PREFIX = "Current terminal state:\n"
CONFIRM_SUFFIX = '\n\nAre you sure you want to mark the task as complete? This will trigger your solution to be graded and you won\'t be able to make any further corrections. If so, include "task_complete": true in your JSON response again.'
# Every terminal capture starts with one of these (``TmuxSession.get_incremental_output``, timeout.txt).
TERMINAL_PREFIXES = ("New Terminal Output:\n", "Current Terminal Screen:\n", "Previous command:\n")
# User turns after an assistant turn that are not main-loop observations.
NOT_OBSERVATION_PREFIXES = (
    "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n",
    "Here are the answers the other agent provided.\n\n",
)


def load_harbor_json_parser(path: Path = HARBOR_JSON_PARSER):
    """Harbor's ``TerminusJSONPlainParser``, loaded from the file (it only needs the stdlib)."""
    name = "_harbor_terminus_json_plain_parser"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None, f"cannot load {path}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].TerminusJSONPlainParser()


@dataclasses.dataclass(frozen=True)
class Turn:
    """How the harness handles one assistant text: kind is "error" (nothing executed), "complete"
    (commands executed, task_complete) or "continue" (commands executed)."""

    kind: str
    feedback: str
    commands: tuple
    complete: bool


def harness_turn(parser, content: str) -> Turn:
    r = parser.parse_response(content)
    feedback = ""  # Terminus2._handle_llm_interaction
    if r.error:
        feedback += f"ERROR: {r.error}"
        if r.warning:
            feedback += f"\nWARNINGS: {r.warning}"
    elif r.warning:
        feedback += f"WARNINGS: {r.warning}"
    commands = tuple((c.keystrokes, min(c.duration, 60)) for c in r.commands)
    if "ERROR:" in feedback:  # Terminus2._run_agent_loop branch order
        kind = "error"
    elif r.is_task_complete:
        kind = "complete"
    else:
        kind = "continue"
    return Turn(kind, feedback, commands, bool(r.is_task_complete))


def harness_observation(turn: Turn, terminal: str | None, pending_completion: bool) -> str | None:
    """The next prompt the harness sends (None: the run ends). ``terminal`` is the terminal part as
    it appears in the observation (already length-limited by the harness)."""
    if turn.kind == "error":
        return f"{PARSE_ERROR_PREFIX}{turn.feedback}{PARSE_ERROR_SUFFIX}"
    if turn.kind == "complete":
        return None if pending_completion else f"{CONFIRM_PREFIX}{terminal}{CONFIRM_SUFFIX}"
    if "WARNINGS:" in turn.feedback:
        return f"{WARNINGS_PREFIX}{turn.feedback}\n\n{terminal}"
    return terminal


def split_observation(content: str) -> tuple[str, str | None, str | None]:
    """(kind, feedback header, terminal part) of a stored observation; kind "unknown" otherwise."""
    if content.startswith(PARSE_ERROR_PREFIX) and content.endswith(PARSE_ERROR_SUFFIX):
        return "error", content[len(PARSE_ERROR_PREFIX) : -len(PARSE_ERROR_SUFFIX)], None
    if content.startswith(CONFIRM_PREFIX) and content.endswith(CONFIRM_SUFFIX):
        return "complete", None, content[len(CONFIRM_PREFIX) : -len(CONFIRM_SUFFIX)]
    if content.startswith(WARNINGS_PREFIX):
        cuts = [content.find("\n\n" + p, len(WARNINGS_PREFIX)) for p in TERMINAL_PREFIXES]
        cuts = [c for c in cuts if c >= 0]
        if cuts:
            cut = min(cuts)
            return "continue", content[len(WARNINGS_PREFIX) : cut], content[cut + 2 :]
    if content.startswith(TERMINAL_PREFIXES):
        return "continue", "", content
    return "unknown", None, None


def _strip_trailing_fence(text: str) -> str:
    stripped = text.rstrip()
    return stripped[:-3] if stripped.endswith("```") else text


def _unfence(text: str) -> str:
    body = _strip_trailing_fence(text).rstrip("\n").lstrip()
    for opener in ("```json\n", "```\n", "```json", "```"):
        if body.startswith(opener):
            return body[len(opener) :]
    return body


def datagen_text_candidates(content: str) -> list[str]:
    """Texts the datagen harness may have parsed: with or without the think block, the trailing
    fence, or the whole code fence."""
    out = []
    k = content.find("</think>")
    for text in (content, content[k + len("</think>") :] if k >= 0 else content):
        out += [text, _strip_trailing_fence(text), _unfence(text)]
    return list(dict.fromkeys(out))


def _stored_header_matches(turn: Turn, kind: str, header: str | None) -> bool:
    if turn.kind != kind:
        return False
    if kind == "complete":
        return True
    if kind == "error":
        return turn.feedback == header
    return header == (turn.feedback if "WARNINGS:" in turn.feedback else "")


def rewrite_observation(parser, assistant: str, observation: str, pending_completion: bool):
    """Returns (new observation or None if unrepairable, reason, new pending_completion).

    The observation is rebuilt with ``harness_observation`` from the stored assistant text. It is
    repairable when what the harness executes is unchanged: same kind, and (unless nothing was
    executed) every candidate datagen parse that reproduces the stored header has the same commands
    and completion flag as the stored text.
    """
    kind, header, terminal = split_observation(observation)
    assert kind != "unknown", f"not a main-loop observation: {observation[:200]!r}"
    turn = harness_turn(parser, assistant)
    if turn.kind != kind:
        return None, f"kind_{kind}_to_{turn.kind}", pending_completion
    if kind == "error":
        return harness_observation(turn, None, pending_completion), "ok", pending_completion
    if kind == "complete" and pending_completion:
        return None, "run_would_have_ended", pending_completion
    candidates = [harness_turn(parser, text) for text in datagen_text_candidates(assistant)]
    matching = [t for t in candidates if _stored_header_matches(t, kind, header)]
    if not matching:
        return None, "no_datagen_parse_reproduces_header", pending_completion
    if any((t.commands, t.complete) != (turn.commands, turn.complete) for t in matching):
        return None, "executed_commands_differ", pending_completion
    return harness_observation(turn, terminal, pending_completion), "ok", kind == "complete"


def _history_key(messages: list[dict], i: int) -> str:
    """Identity of (rendered context, target) of assistant turn ``i``.

    The chat template renders a context from the roles, the user/system contents and the
    think-stripped assistant contents only, so these determine the prompt string (and its tokens).
    """
    h = hashlib.sha256()
    for m in messages[:i]:
        content = m["content"]
        if m["role"] == "assistant" and "</think>" in content:
            content = content.split("</think>")[-1].lstrip("\n")
        h.update(json.dumps([m["role"], content]).encode())
    h.update(json.dumps(["target", messages[i]["content"]]).encode())
    return h.hexdigest()


def _rewrite_observations(parser, messages: list[dict], stats: collections.Counter) -> None:
    """Recompute the main-loop observation headers in place; unsupervise turns after an unrepairable one."""
    pending, broken = False, False
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if broken and m.get("loss", True):
                m["loss"] = False
                stats["turns_after_unreproducible_unsupervised"] += 1
            continue
        if i == 0 or messages[i - 1]["role"] != "assistant":
            continue
        if m["content"].startswith(NOT_OBSERVATION_PREFIXES):
            pending = False  # a handoff starts a new main loop (we assume no pending completion)
            continue
        if split_observation(m["content"])[0] == "unknown":
            # no harness observation has this shape (none in the OTA snapshot; the tests assert it)
            stats["user_after_assistant_not_observation"] += 1
            continue
        stats["observations_checked"] += 1
        if broken:
            continue
        new, reason, pending = rewrite_observation(parser, messages[i - 1]["content"], m["content"], pending)
        if new is None:
            broken = True
            stats[f"unreproducible_{reason}"] += 1
            stats["unreproducible_observations"] += 1
            continue
        stats["observations_rewritten"] += new != m["content"]
        m["content"] = new


def is_main_loop_turn(messages: list[dict], i: int) -> bool:
    """Assistant turn ``i`` answers the initial prompt, a harness observation, or a handoff."""
    if messages[i]["role"] != "assistant" or i == 0 or messages[i - 1]["role"] != "user":
        return False
    prev = messages[i - 1]["content"]
    if i == 1:  # the terminus-2 initial prompt (template: "...Task Description:\n{instruction}\n\nCurrent terminal state:\n...")
        return "\nTask Description:\n" in prev and "\n\nCurrent terminal state:\n" in prev
    if messages[i - 2]["role"] != "assistant":
        return False  # question prompt, or the summary request after the pending observation
    if prev.startswith(NOT_OBSERVATION_PREFIXES[1]):
        return True  # the handoff prompt
    return split_observation(prev)[0] != "unknown"


def _unsupervise_parse_failures(parser, messages: list[dict], stats: collections.Counter) -> None:
    for i, m in enumerate(messages):
        if not is_main_loop_turn(messages, i) or harness_turn(parser, m["content"]).kind != "error":
            continue
        stats["main_loop_targets_failing_parse"] += 1
        if m.get("loss", True):
            m["loss"] = False
            stats["main_loop_targets_failing_parse_unsupervised"] += 1


def convert_rows(rows: list[dict], parser=None) -> tuple[list[dict], collections.Counter]:
    parser = parser or load_harbor_json_parser()
    stats = collections.Counter()
    seen: set[str] = set()
    out = []
    for row_index, row in enumerate(rows):
        messages, warned = [], False
        for m in row["conversations"]:
            role, content = m["role"], m["content"]
            assert role in ("user", "assistant"), f"row {row_index}: role {role!r}"
            msg = {"role": role, "content": content}
            if role == "user":
                normalized = normalize_system_prompt(content)
                stats["system_prompt_normalized"] += normalized != content
                msg["content"], n = strip_system_warnings(normalized)
                stats["warnings_stripped"] += n
                warned = warned or n > 0  # any warned message since the last assistant turn
            else:
                if warned:
                    msg["loss"] = False
                    stats["turns_after_warning_unsupervised"] += 1
                warned = False
            messages.append(msg)
        _rewrite_observations(parser, messages, stats)
        _unsupervise_parse_failures(parser, messages, stats)
        # a trailing user message has no target; inference never ends a trace on one either
        while messages and messages[-1]["role"] == "user":
            messages.pop()
            stats["trailing_user_dropped"] += 1
        assert messages and messages[0]["role"] == "user", f"row {row_index}: must start with a user message"
        for i, m in enumerate(messages):
            if m["role"] != "assistant" or m.get("loss") is False:
                continue
            key = _history_key(messages, i)
            if key in seen:
                m["loss"] = False
                stats["duplicate_turns_unsupervised"] += 1
            seen.add(key)
        stats["rows"] += 1
        stats["assistant_turns"] += sum(m["role"] == "assistant" for m in messages)
        stats["supervised_turns"] += sum(m["role"] == "assistant" and m.get("loss", True) for m in messages)
        roles = [m["role"] for m in messages]
        stats["non_alternating_rows"] += roles != ["user", "assistant"] * (len(roles) // 2)
        out.append(
            {
                "messages": messages,
                "metadata": {"task": row["task"], "trace_source": row["trace_source"], "row_index": row_index},
            }
        )
    return out, stats


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(args.src.rglob("*.parquet"))
    rows = pq.read_table(shards).to_pylist()
    out, stats = convert_rows(rows)
    with args.dst.open("w") as f:
        for record in out:
            f.write(json.dumps(record) + "\n")
    print(f"{len(shards)} shards -> {args.dst}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
