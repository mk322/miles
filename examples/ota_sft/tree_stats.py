"""Token statistics of ``--sft-tree-attention`` packing on an OTA jsonl (and optional per-turn verification).

For every row: tokens of the current verbatim pipeline (``ota_sft_rollout.encode_conversation``),
of the expanded per-turn samples, and of the trees; dropped turns; prefix breaks (with the tokens
at the break, so a merge across the generation prompt is visible).

Usage (inside the Miles image):
  python examples/ota_sft/tree_stats.py --data /root/datasets/ota_sft_10k.jsonl \
      --tokenizer /root/models/Qwen3-8B --chat-template /path/to/chat_template.jinja [--verify-every 10]
"""

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import asdict

from transformers import AutoTokenizer

from examples.ota_sft.ota_sft_rollout import TARGET_SUFFIX, encode_conversation
from miles.rollout.sft_tree import build_trees, build_turn_samples

_TOK = None
_ARGS = None


def _init(args):
    global _TOK, _ARGS
    _ARGS = args
    _TOK = AutoTokenizer.from_pretrained(args.tokenizer)


def _row(item):
    index, line = item
    messages = json.loads(line)["messages"]
    t0 = time.perf_counter()
    turns = build_turn_samples(_TOK, messages, target_suffix=TARGET_SUFFIX, chat_template=_ARGS.chat_template)
    trees, stats = build_trees(turns, max_context_len=_ARGS.cutoff, max_tree_tokens=_ARGS.max_tree_tokens)
    build_s = time.perf_counter() - t0
    out = {"index": index, "stats": asdict(stats), "build_s": build_s}
    out["trees"] = [len(t.tokens) for t in trees]
    if all(m["role"] in ("user", "assistant") for m in messages) and [m["role"] for m in messages] == [
        "user",
        "assistant",
    ] * (len(messages) // 2):
        ids, mask = encode_conversation(_TOK, messages, _ARGS.cutoff)
        out["verbatim_tokens"], out["verbatim_targets"] = len(ids), sum(mask)
    breaks = []
    for a, b in zip(turns[:-1], turns[1:], strict=True):
        n = len(a.prompt_ids)
        if b.prompt_ids[:n] != a.prompt_ids:
            j = next(i for i in range(n) if i >= len(b.prompt_ids) or a.prompt_ids[i] != b.prompt_ids[i])
            breaks.append({"turn": a.message_index, "at": j, "prompt_k": _TOK.decode(a.prompt_ids[j - 3 : j + 3]),
                           "next": _TOK.decode(b.prompt_ids[j - 3 : j + 3])})
    out["breaks"] = breaks
    if _ARGS.verify_every and index % _ARGS.verify_every == 0:
        for t in turns:
            ref = _TOK.apply_chat_template(
                [{"role": m["role"], "content": m["content"]} for m in messages[: t.message_index]],
                tokenize=True, add_generation_prompt=True, chat_template=_ARGS.chat_template,
            )
            ref = list(ref["input_ids"] if hasattr(ref, "keys") else ref)
            assert ref == list(t.prompt_ids), f"row {index} turn {t.message_index}: prompt != apply_chat_template"
        out["verified_turns"] = len(turns)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--chat-template", required=True)
    p.add_argument("--cutoff", type=int, default=32768)
    p.add_argument("--max-tree-tokens", type=int, default=32768)
    p.add_argument("--verify-every", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    args.chat_template = open(args.chat_template).read()

    with open(args.data) as f:
        lines = list(enumerate(f))
    if args.limit:
        lines = lines[: args.limit]
    with mp.Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        rows = pool.map(_row, lines, chunksize=8)

    agg = Counter()
    for r in rows:
        agg.update({k: v for k, v in r["stats"].items()})
        agg["verbatim_tokens"] += r.get("verbatim_tokens", 0)
        agg["verbatim_targets"] += r.get("verbatim_targets", 0)
        agg["rows_with_breaks"] += bool(r["breaks"])
        agg["rows_without_trees"] += not r["trees"]
        agg["rows_with_dropped_turns"] += r["stats"]["num_dropped_turns"] > 0
        agg["verified_turns"] += r.get("verified_turns", 0)
    n = len(rows)
    summary = {
        "rows": n,
        **agg,
        "tree_over_verbatim": agg["tree_tokens"] / max(agg["verbatim_tokens"], 1),
        "expanded_over_verbatim": agg["expanded_tokens"] / max(agg["verbatim_tokens"], 1),
        "tree_over_expanded": agg["tree_tokens"] / max(agg["expanded_tokens"], 1),
        "mean_tokens_per_row": {k: agg[k] / n for k in ("verbatim_tokens", "expanded_tokens", "tree_tokens")},
        "mean_trees_per_row": agg["num_trees"] / n,
        "max_tree_tokens_seen": max((max(r["trees"]) for r in rows if r["trees"]), default=0),
        "mean_build_s_per_row": sum(r["build_s"] for r in rows) / n,
        "break_examples": [b for r in rows for b in r["breaks"]][:10],
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f)


if __name__ == "__main__":
    main()
