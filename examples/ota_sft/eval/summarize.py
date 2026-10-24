"""Summarize a Harbor job the way OpenThoughts-Agent reports it.

Each task runs n_attempts times; attempt k across all tasks forms run k. The score is the mean
pass@1 over runs, with the standard error of the run means (paper, App. G). Errored trials count
as reward 0 (Harbor mean).

Usage: python examples/ota_sft/eval/summarize.py <job_dir> [<job_dir> ...]
"""

import collections
import glob
import json
import statistics
import sys
from pathlib import Path


def _rewards_by_task(job_dir: Path) -> dict[str, list[float]]:
    by_task = collections.defaultdict(list)
    trials = []
    for f in glob.glob(str(job_dir / "*" / "result.json")):
        r = json.load(open(f))
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward") or 0.0
        trials.append((r.get("started_at") or "", r["task_name"], float(reward)))
    for _, task, reward in sorted(trials):
        by_task[task].append(reward)
    return by_task


def summarize(job_dir: Path) -> str:
    by_task = _rewards_by_task(job_dir)
    n_runs = min(len(v) for v in by_task.values())
    run_means = [100 * statistics.mean(v[k] for v in by_task.values()) for k in range(n_runs)]
    se = statistics.stdev(run_means) / len(run_means) ** 0.5 if n_runs > 1 else float("nan")
    all_mean = 100 * statistics.mean(x for v in by_task.values() for x in v)
    solved_any = sum(1 for v in by_task.values() if max(v) > 0)
    return (
        f"{job_dir.name}: tasks={len(by_task)} runs={n_runs} run_means={[round(m, 2) for m in run_means]} "
        f"mean={statistics.mean(run_means):.2f} ± {se:.2f} (all-trial mean {all_mean:.2f}) "
        f"pass@{n_runs}={100 * solved_any / len(by_task):.1f}"
    )


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(summarize(Path(d)))
