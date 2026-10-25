# Datagen-parity eval config

`terminus2_local_docker_parity.yaml` is `terminus2_local_docker.yaml` plus two agent
kwargs:

- `datagen_parity: true`: terminus-2 sends the model the same prompts as the
  2026-04 harness that generated OpenThoughts-Agent-SFT-10K. You need the harbor-marin
  branch `datagen-parity` (github.com/mk322/harbor) checked out at `$HARBOR_SRC` and installed
  editable into `$OTA_ROOT/venvs/harbor`.
- `proactive_summarization_threshold: 8000`: the datagen default. The base yaml uses
  2048.

With the switch on, the model sees:

- a summary request without the "state was captured before any history was trimmed"
  appendix; its history ends `assistant, user(pending observation), user(summary
  request)`, the shape of all 381 summary and 314 answers rows in the SFT data;
- the datagen-era overflow ladder: unwind to 4000 free tokens, then the full summary,
  then "Briefly continue this task: ...", then "{instruction}\n\nCurrent state:
  ...". A proactive summary that fails is skipped, and the turn continues
  unsummarized;
- summary subcalls with no `max_tokens` and no client-side history trimming;
- per-command tmux execution, and the datagen-era 2000-line pane scrollback with
  unbounded capture (this affects when an observation becomes `Current Terminal
  Screen:`);
- requests always rendered from messages (no token-in/token-out continuation).

`interleaved_thinking` cannot be combined with `datagen_parity`. The system prompt,
observation formats, warnings, completion confirmation, and the question, answers and
handoff prompts were already identical, and they are unchanged.

`run_eval.sh` reads `terminus2_local_docker.yaml` unless `CONFIG` names another yaml. To use the parity config,
run `CONFIG=$PWD/terminus2_local_docker_parity.yaml ./run_eval.sh ...`, or queue checkpoints with `eval_queue.py --priority-queue TAG:RUN_DIR:terminus2_local_docker_parity.yaml:STEPS`. Alternatively,
pass the two kwargs through it:

```bash
AGENT_KWARGS=$'datagen_parity=true\nproactive_summarization_threshold=8000' \
  ./run_eval.sh tb2 <served_model_name>
```

Tests: `tests/unit/agents/terminus_2/test_terminus_2_datagen_parity_*.py` in harbor-marin.
They diff whole runs against verbatim copies of the 612aaedd code, and with the switch
off against the pre-switch `main`. They also rebuild every summary, answers and
post-handoff row of the dataset byte for byte.
