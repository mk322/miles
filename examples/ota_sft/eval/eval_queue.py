"""Evaluate queues of SFT checkpoints (SWE-V-100 + TB2, 3 attempts) on a set of GPUs.

Each GPU worker repeatedly takes the next checkpoint whose HF export is complete, serves it with
vLLM (OpenThoughts-Agent's SFT-model serving: no tool/reasoning parser, prefix caching, 32K), runs
both Harbor jobs through run_eval.sh, then stops the server and moves on immediately (so no other
user's scheduler grabs the idle GPU). A job directory that already exists but is unfinished is
resumed with `harbor jobs resume` at the new concurrency; finished jobs are skipped.

The base queue (--tag/--run-dir/--steps/--config) can be preceded by priority queues
(--priority-queue TAG:RUN_DIR:CONFIG:STEPS, highest first). A checkpoint of a higher queue is always
taken before any checkpoint of a lower one, and when it becomes available while every GPU is busy,
one worker running a lower-queue checkpoint preempts it: its Harbor jobs get SIGINT (unfinished trials
end as CancelledError, which `harbor jobs resume` removes and reruns), its server is stopped, and the
checkpoint goes back to its place in the queue.

OTA_ROOT (or --ota-root) holds eval_jobs/, logs/, venvs/harbor and the datasets run_eval.sh reads.

Usage:
  python3 examples/ota_sft/eval/eval_queue.py --gpus 0,1 --run-dir $OTA_ROOT/runs/ota_lora_r64_lr4e-4 \
      --steps 664,569,474,379,299,284,189,94
  python3 examples/ota_sft/eval/eval_queue.py --gpus 0,1 --run-dir $OTA_ROOT/runs/ota_lora_r64_lr4e-4 \
      --steps 664,569,474,379,299,284,189,94 \
      --priority-queue full_tree:$OTA_ROOT/runs/ota_full_tree_tp2:terminus2_local_docker_parity.yaml:728,727,623,519,415,311,207,103
"""

import argparse
import getpass
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VLLM_IMAGE = "vllm/vllm-openai:v0.20.0"
_BENCHES = ("swebench100", "tb2")
_DEFAULT_TAG = "lora_r64"


def _log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


@dataclass(frozen=True)
class Entry:
    tag: str
    step: int
    run_dir: Path
    config: str  # agent yaml next to this file
    group: int  # queue index, 0 = highest priority
    rank: int  # position within its queue

    @property
    def key(self) -> str:
        # the default queue keeps the names of the jobs, claims and servers it created before queues had tags
        return f"step{self.step}" if self.tag == _DEFAULT_TAG else f"{self.tag}_step{self.step}"

    @property
    def served_name(self) -> str:
        return f"ota-{self.tag.replace('_', '-')}-step{self.step}"

    @property
    def container(self) -> str:
        return f"ota-vllm-sft-{self.key.replace('_', '-')}"

    @property
    def default_port(self) -> int:
        return (9000 if self.tag == _DEFAULT_TAG else 10000) + self.step

    def job_name(self, bench: str) -> str:
        return f"sft_{self.tag}_step{self.step}_{bench}_k3"

    def hf_dir(self) -> Path:
        return self.run_dir / f"hf/step_{self.step}"

    def available(self) -> bool:
        return (self.hf_dir() / ".complete").exists()


def _parse_queue(spec: str, group: int) -> list[Entry]:
    tag, run_dir, config, steps = spec.split(":")
    if not (_HERE / config).exists():
        raise FileNotFoundError(_HERE / config)
    return [Entry(tag, int(s), Path(run_dir), config, group, i) for i, s in enumerate(steps.split(","))]


def _pick(queue: list[Entry], claim, below_group: int | None = None) -> Entry | None:
    """Remove and return the first available entry of `queue` (kept in priority order) that `claim`
    accepts, considering only groups < below_group when given. Entries whose claim fails belong to
    another queue process and are dropped."""
    for e in [e for e in queue if (below_group is None or e.group < below_group) and e.available()]:
        queue.remove(e)
        if claim(e):
            return e
    return None


def _requeue(queue: list[Entry], e: Entry) -> None:
    queue.append(e)
    queue.sort(key=lambda x: (x.group, x.rank))


def _finished(job_dir: Path) -> bool:
    try:
        return bool(json.loads((job_dir / "result.json").read_text()).get("finished_at"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _pids(job: str) -> list[int]:
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"harbor jobs .*{job}"], capture_output=True, text=True)
    return [int(p) for p in out.stdout.split()]


def _active(job: str) -> bool:
    return bool(_pids(job))


def _claim_path(jobs_dir: Path, e: Entry) -> Path:
    return jobs_dir / f".queue_claim_{e.key}"


def _claim(jobs_dir: Path, e: Entry) -> bool:
    try:
        os.close(os.open(_claim_path(jobs_dir, e), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _port_for(e: Entry, jobs_dir: Path) -> int:
    # A resumed job must be served on the port its config.json already points at.
    for bench in _BENCHES:
        cfg = jobs_dir / e.job_name(bench) / "config.json"
        if cfg.exists():
            m = re.search(r":(\d+)/v1", json.loads(cfg.read_text())["agents"][0]["kwargs"]["api_base"])
            if m:
                return int(m.group(1))
    return e.default_port


def _start_vllm(e: Entry, gpu: str, port: int, mem_util: float) -> str:
    name = e.container
    running = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{name}$", "-f", "status=running"],
                             capture_output=True, text=True).stdout.strip()
    if running and subprocess.run(["curl", "-sf", f"localhost:{port}/v1/models"], capture_output=True).returncode == 0:
        return name  # adopt the server that is already up
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run([
        "docker", "run", "-d", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={getpass.getuser()}", "-e", f"LOGNAME={getpass.getuser()}", "-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "--ipc=host",
        "--gpus", f'"device={gpu}"', "-p", f"{port}:{port}", "-v", f"{e.hf_dir()}:/model:ro", _VLLM_IMAGE, "/model",
        "--served-model-name", e.served_name, "--host", "0.0.0.0", "--port", str(port),
        "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(mem_util), "--disable-custom-all-reduce",
        "--max-model-len", "32768", "--trust-remote-code", "--enable-prefix-caching",
    ], check=True, capture_output=True)
    for _ in range(90):
        if subprocess.run(["curl", "-sf", f"localhost:{port}/v1/models"], capture_output=True).returncode == 0:
            return name
        time.sleep(10)
    raise RuntimeError(f"vLLM for {e.key} did not come up on GPU {gpu}")


def _run_job(e: Entry, bench: str, port: int, n: int, ota_root: Path) -> subprocess.Popen:
    job = e.job_name(bench)
    job_dir = ota_root / "eval_jobs" / job
    log = open(ota_root / "logs" / f"queue_{job}.log", "a")
    if job_dir.exists():
        cfg = json.loads((job_dir / "config.json").read_text())
        cfg["n_concurrent_trials"] = n
        (job_dir / "config.json").write_text(json.dumps(cfg, indent=4))
        cmd = [str(ota_root / "venvs/harbor/bin/harbor"), "jobs", "resume", "-p", str(job_dir)]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    env = dict(os.environ, API_BASE=f"http://localhost:{port}/v1", JOB_NAME=job, CONFIG=str(_HERE / e.config))
    env.pop("AGENT_KWARGS", None)  # SFT checkpoints are served without the base model's thinking kwarg
    cmd = [str(_HERE / "run_eval.sh"), "swebench" if bench == "swebench100" else "tb2", e.served_name, "-n", str(n)]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def _stop_jobs(e: Entry) -> None:
    jobs = [e.job_name(b) for b in _BENCHES]
    for sig, wait_s in ((signal.SIGINT, 900), (signal.SIGKILL, 60)):
        for job in jobs:
            for pid in _pids(job):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.time() + wait_s
        while time.time() < deadline and any(_active(j) for j in jobs):
            time.sleep(10)
        if not any(_active(j) for j in jobs):
            return


def _worker(gpu: str, queue: list[Entry], lock: threading.Lock, args) -> None:
    jobs_dir = args.ota_root / "eval_jobs"
    claim = lambda e: _claim(jobs_dir, e)  # noqa: E731
    e = None
    while True:
        if e is None:
            with lock:
                e = _pick(queue, claim)
                if e is None and not queue:
                    _log(f"GPU {gpu}: queue empty, done")
                    return
            if e is None:
                time.sleep(300)
                continue
        if all(_finished(jobs_dir / e.job_name(b)) for b in _BENCHES):
            _log(f"GPU {gpu}: {e.key} already evaluated, skipping")
            e = None
            continue
        port = _port_for(e, jobs_dir)
        _log(f"GPU {gpu}: serving {e.key} ({e.config}) on :{port}")
        name = _start_vllm(e, gpu, port, args.gpu_mem_util)
        # jobs that are already running (e.g. adopted after a restart of this script) are waited on, not relaunched
        for b in _BENCHES:
            if not _finished(jobs_dir / e.job_name(b)) and not _active(e.job_name(b)):
                _run_job(e, b, port, args.n, args.ota_root)
        time.sleep(30)
        preempted_by = None
        while any(_active(e.job_name(b)) for b in _BENCHES):
            time.sleep(60)
            if e.group > 0:
                with lock:
                    preempted_by = _pick(queue, claim, below_group=e.group)
                if preempted_by is not None:
                    break
        if preempted_by is not None:
            _log(f"GPU {gpu}: pausing {e.key} for {preempted_by.key}")
            _stop_jobs(e)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            with lock:
                _claim_path(jobs_dir, e).unlink(missing_ok=True)
                _requeue(queue, e)
            e = preempted_by
            continue
        _log(f"GPU {gpu}: {e.key} done")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        e = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--steps", required=True, help="checkpoint steps in priority order")
    parser.add_argument("--tag", default=_DEFAULT_TAG, help="job names are sft_<tag>_step<N>_<bench>_k3")
    parser.add_argument("--config", default="terminus2_local_docker.yaml", help="agent yaml next to this file")
    parser.add_argument("--priority-queue", action="append", default=[], metavar="TAG:RUN_DIR:CONFIG:STEPS",
                        help="evaluated before, and preempting, the base queue; repeat, highest first")
    parser.add_argument("--n", type=int, default=8, help="concurrent trials per job")
    parser.add_argument("--gpu-mem-util", type=float, default=0.92, help="vLLM --gpu-memory-utilization")
    parser.add_argument("--ota-root", type=Path, default=os.environ.get("OTA_ROOT"), required="OTA_ROOT" not in os.environ)
    parser.add_argument("--run-dir", type=Path, required=True, help="run directory of the base queue (holds hf/step_<N>)")
    args = parser.parse_args()

    specs = args.priority_queue + [f"{args.tag}:{args.run_dir}:{args.config}:{args.steps}"]
    queue = [e for g, spec in enumerate(specs) for e in _parse_queue(spec, g)]
    lock = threading.Lock()
    threads = [threading.Thread(target=_worker, args=(g, queue, lock, args)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
        time.sleep(5)  # workers pick in --gpus order, so an adopted server stays on the GPU it runs on
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
