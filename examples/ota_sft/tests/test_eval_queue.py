"""Tests for ``eval/eval_queue.py``: names, priority order, claims and preemption (no GPU, docker or Harbor).

Run: python -m pytest examples/ota_sft/tests/test_eval_queue.py -q
"""

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location("eval_queue", Path(__file__).resolve().parents[1] / "eval/eval_queue.py")
eq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eq)

_real_sleep = time.sleep
PARITY = "terminus2_local_docker_parity.yaml"
BASE = "terminus2_local_docker.yaml"


def _make_run(tmp_path: Path, name: str, steps_done: list[int]) -> Path:
    run = tmp_path / name
    for s in steps_done:
        (run / f"hf/step_{s}").mkdir(parents=True)
        (run / f"hf/step_{s}/.complete").touch()
    return run


def _queue(tmp_path, full_done, lora_done, full_steps="207,103", lora_steps="664,474,379"):
    full = _make_run(tmp_path, "full", full_done)
    lora = _make_run(tmp_path, "lora", lora_done)
    return eq._parse_queue(f"full_tree:{full}:{PARITY}:{full_steps}", 0) + eq._parse_queue(
        f"lora_r64:{lora}:{BASE}:{lora_steps}", 1
    )


def test_default_tag_keeps_legacy_names():
    e = eq.Entry("lora_r64", 474, Path("/r"), BASE, 1, 0)
    assert e.job_name("tb2") == "sft_lora_r64_step474_tb2_k3"
    assert e.served_name == "ota-lora-r64-step474"
    assert e.container == "ota-vllm-sft-step474"
    assert e.key == "step474"
    assert e.default_port == 9474


def test_tagged_queue_names_do_not_collide_with_legacy():
    e = eq.Entry("full_tree", 103, Path("/r"), PARITY, 0, 0)
    assert e.job_name("swebench100") == "sft_full_tree_step103_swebench100_k3"
    assert e.served_name == "ota-full-tree-step103"
    assert e.container == "ota-vllm-sft-full-tree-step103"
    assert e.key == "full_tree_step103"
    assert e.default_port == 10103
    assert e.hf_dir() == Path("/r/hf/step_103")


def test_parse_queue_rejects_missing_config():
    with pytest.raises(FileNotFoundError):
        eq._parse_queue("full_tree:/r:no_such.yaml:103", 0)


def test_parse_queue_order_and_groups(tmp_path):
    q = _queue(tmp_path, [], [])
    assert [(e.tag, e.step, e.group, e.rank) for e in q] == [
        ("full_tree", 207, 0, 0), ("full_tree", 103, 0, 1),
        ("lora_r64", 664, 1, 0), ("lora_r64", 474, 1, 1), ("lora_r64", 379, 1, 2),
    ]
    assert all(e.config == PARITY for e in q if e.tag == "full_tree")


def test_pick_takes_highest_available(tmp_path):
    q = _queue(tmp_path, full_done=[103], lora_done=[664, 474])
    e = eq._pick(q, lambda e: True)
    assert (e.tag, e.step) == ("full_tree", 103)  # 207 not exported yet, but the full queue still comes first
    e = eq._pick(q, lambda e: True)
    assert (e.tag, e.step) == ("lora_r64", 664)
    assert [x.step for x in q] == [207, 474, 379]


def test_pick_below_group_only_considers_higher_queues(tmp_path):
    q = _queue(tmp_path, full_done=[], lora_done=[664])
    assert eq._pick(q, lambda e: True, below_group=1) is None
    assert len(q) == 5
    (tmp_path / "full/hf/step_207").mkdir(parents=True)
    (tmp_path / "full/hf/step_207/.complete").touch()
    e = eq._pick(q, lambda e: True, below_group=1)
    assert (e.tag, e.step) == ("full_tree", 207)
    assert eq._pick(q, lambda e: True, below_group=0) is None


def test_pick_drops_entries_claimed_elsewhere(tmp_path):
    q = _queue(tmp_path, full_done=[103], lora_done=[664])
    e = eq._pick(q, lambda e: e.tag != "full_tree")
    assert (e.tag, e.step) == ("lora_r64", 664)
    assert all(not (x.tag == "full_tree" and x.step == 103) for x in q)


def test_pick_waits_for_complete_marker(tmp_path):
    q = _queue(tmp_path, full_done=[], lora_done=[])
    (tmp_path / "full/hf/step_103").mkdir(parents=True)  # export still being written
    assert eq._pick(q, lambda e: True) is None


def test_requeue_restores_priority_position(tmp_path):
    q = _queue(tmp_path, full_done=[], lora_done=[664, 474, 379])
    e = eq._pick(q, lambda e: True)
    assert e.step == 664
    eq._requeue(q, e)
    assert [(x.group, x.rank) for x in q] == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]


def test_claim_is_exclusive_and_keyed(tmp_path):
    a = eq.Entry("lora_r64", 379, Path("/r"), BASE, 1, 0)
    b = eq.Entry("full_tree", 379, Path("/r"), PARITY, 0, 0)
    assert eq._claim(tmp_path, a) and not eq._claim(tmp_path, a)
    assert eq._claim(tmp_path, b)  # same step number, different queue
    assert (tmp_path / ".queue_claim_step379").exists()
    assert (tmp_path / ".queue_claim_full_tree_step379").exists()


def test_port_for_resumed_job_reads_config(tmp_path):
    e = eq.Entry("lora_r64", 379, Path("/r"), BASE, 1, 0)
    assert eq._port_for(e, tmp_path) == 9379
    d = tmp_path / e.job_name("tb2")
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"agents": [{"kwargs": {"api_base": "http://localhost:8002/v1"}}]}))
    assert eq._port_for(e, tmp_path) == 8002


class _Sim:
    """Fakes for docker, Harbor and the clock. A job runs until `finish` marks it done or it is stopped."""

    def __init__(self, monkeypatch, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.running: set[str] = set()
        self.events: list[tuple] = []
        self.ticks = 0
        self.on_tick = lambda sim: None
        self._clock = threading.Lock()  # workers in several threads share one simulated clock
        monkeypatch.setattr(eq, "_start_vllm", lambda e, gpu, port, mu: self.events.append(("serve", gpu, e.key, port)) or e.container)
        monkeypatch.setattr(eq, "_run_job", self._run_job)
        monkeypatch.setattr(eq, "_active", lambda job: job in self.running)
        monkeypatch.setattr(eq, "_stop_jobs", self._stop)
        monkeypatch.setattr(eq.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
        monkeypatch.setattr(eq.time, "sleep", self._sleep)

    def _run_job(self, e, bench, port, n, root):
        self.events.append(("job", e.job_name(bench), e.config, port))
        self.running.add(e.job_name(bench))

    def _stop(self, e):
        self.events.append(("stop", e.key))
        for b in eq._BENCHES:
            self.running.discard(e.job_name(b))

    def _sleep(self, s):
        _real_sleep(0.0005)  # let the other worker thread run, as a real sleep would
        with self._clock:
            self.ticks += 1
            if self.ticks > 5000:
                raise RuntimeError("simulation did not terminate")
            self.on_tick(self)

    def finish(self, e):
        for b in eq._BENCHES:
            d = self.jobs_dir / e.job_name(b)
            d.mkdir(exist_ok=True)
            (d / "result.json").write_text(json.dumps({"finished_at": "x"}))
            self.running.discard(e.job_name(b))


def _args(tmp_path):
    (tmp_path / "eval_jobs").mkdir(exist_ok=True)
    return SimpleNamespace(ota_root=tmp_path, n=8, gpu_mem_util=0.92)


def test_worker_preempts_lora_for_new_full_checkpoint(tmp_path, monkeypatch):
    args = _args(tmp_path)
    q = _queue(tmp_path, full_done=[], lora_done=[474], full_steps="103", lora_steps="474")
    sim = _Sim(monkeypatch, tmp_path / "eval_jobs")
    full103, lora474 = q[0], q[1]

    def tick(sim):
        if sim.ticks == 5:  # the full-FT export lands while step 474 is being evaluated
            (tmp_path / "full/hf/step_103").mkdir(parents=True)
            (tmp_path / "full/hf/step_103/.complete").touch()
        if sim.ticks == 20:
            sim.finish(full103)
        if sim.ticks == 40:
            sim.finish(lora474)

    sim.on_tick = tick
    eq._worker("0", q, threading.Lock(), args)
    kinds = [(ev[0], ev[2] if ev[0] == "serve" else ev[1]) for ev in sim.events]
    assert kinds == [
        ("serve", "step474"),
        ("job", "sft_lora_r64_step474_swebench100_k3"), ("job", "sft_lora_r64_step474_tb2_k3"),
        ("stop", "step474"),
        ("serve", "full_tree_step103"),
        ("job", "sft_full_tree_step103_swebench100_k3"), ("job", "sft_full_tree_step103_tb2_k3"),
        ("serve", "step474"),  # resumed after the full-FT eval
        ("job", "sft_lora_r64_step474_swebench100_k3"), ("job", "sft_lora_r64_step474_tb2_k3"),
    ]
    full_jobs = [ev for ev in sim.events if ev[0] == "job" and "full_tree" in ev[1]]
    assert all(ev[2] == PARITY and ev[3] == 10103 for ev in full_jobs)
    assert all(ev[2] == BASE for ev in sim.events if ev[0] == "job" and "lora" in ev[1])
    assert not q
    # the paused step's claim was released and taken again; the full step's claim is held
    assert (tmp_path / "eval_jobs/.queue_claim_step474").exists()
    assert (tmp_path / "eval_jobs/.queue_claim_full_tree_step103").exists()


def test_worker_never_preempts_within_the_top_queue(tmp_path, monkeypatch):
    args = _args(tmp_path)
    q = _queue(tmp_path, full_done=[103], lora_done=[], full_steps="207,103", lora_steps="474")
    sim = _Sim(monkeypatch, tmp_path / "eval_jobs")
    full207, full103 = q[0], q[1]

    def tick(sim):
        if sim.ticks == 5:  # a newer full checkpoint appears while an older one is evaluated
            (tmp_path / "full/hf/step_207").mkdir(parents=True)
            (tmp_path / "full/hf/step_207/.complete").touch()
        if sim.ticks == 20:
            sim.finish(full103)
        if sim.ticks == 30:
            sim.finish(full207)
        if sim.ticks == 31:
            (tmp_path / "lora/hf/step_474").mkdir(parents=True)
            (tmp_path / "lora/hf/step_474/.complete").touch()
        if sim.ticks == 50:
            sim.finish(q_lora[0])

    q_lora = [e for e in q if e.tag == "lora_r64"]
    sim.on_tick = tick
    eq._worker("1", q, threading.Lock(), args)
    assert not any(ev[0] == "stop" for ev in sim.events)
    assert [ev[2] for ev in sim.events if ev[0] == "serve"] == ["full_tree_step103", "full_tree_step207", "step474"]


def test_two_workers_preempt_only_once(tmp_path, monkeypatch):
    args = _args(tmp_path)
    q = _queue(tmp_path, full_done=[], lora_done=[474, 379], full_steps="103", lora_steps="474,379")
    sim = _Sim(monkeypatch, tmp_path / "eval_jobs")
    lock = threading.Lock()
    by_key = {e.key: e for e in q}
    done = threading.Event()

    def tick(sim):
        # the export lands once both GPUs are busy with LoRA steps, so exactly one of them must yield
        serving = {ev[2] for ev in sim.events if ev[0] == "serve"}
        if {"step474", "step379"} <= serving and not (tmp_path / "full/hf/step_103").exists():
            (tmp_path / "full/hf/step_103").mkdir(parents=True)
            (tmp_path / "full/hf/step_103/.complete").touch()
            sim.export_tick = sim.ticks
        if getattr(sim, "export_tick", None) is not None and sim.ticks == sim.export_tick + 50:
            for k in ("full_tree_step103", "step474", "step379"):
                sim.finish(by_key[k])
            done.set()

    sim.on_tick = tick
    ts = [threading.Thread(target=eq._worker, args=(g, q, lock, args)) for g in ("0", "1")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert done.is_set() and not any(t.is_alive() for t in ts), (sim.ticks, sim.events)
    assert len([ev for ev in sim.events if ev[0] == "stop"]) == 1
    assert len([ev for ev in sim.events if ev[0] == "serve" and ev[2] == "full_tree_step103"]) == 1
