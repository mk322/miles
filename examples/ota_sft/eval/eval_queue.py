"""Evaluate a queue of SFT checkpoints (SWE-V-100 + TB2, 3 attempts) on a set of GPUs.

Each GPU worker repeatedly takes the next checkpoint whose HF export is complete, serves it with
vLLM (OpenThoughts-Agent's SFT-model serving: no tool/reasoning parser, prefix caching, 32K), runs
both Harbor jobs through run_eval.sh, then stops the server and moves on immediately (so no other
user's scheduler grabs the idle GPU). A job directory that already exists but is unfinished is
resumed with `harbor jobs resume` at the new concurrency; finished jobs are skipped.

Usage:
  sg docker -c "python3 examples/ota_sft/eval/eval_queue.py --gpus 0,1 --steps 664,569,474,379,299,284,189,94"
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VLLM_IMAGE = "vllm/vllm-openai:v0.20.0"
_BENCHES = ("swebench100", "tb2")


def _log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def _job_name(step: int, bench: str) -> str:
    return f"sft_lora_r64_step{step}_{bench}_k3"


def _finished(job_dir: Path) -> bool:
    try:
        return bool(json.loads((job_dir / "result.json").read_text()).get("finished_at"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _active(job: str) -> bool:
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"harbor jobs .*{job}"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def _claim(jobs_dir: Path, step: int) -> bool:
    try:
        os.close(os.open(jobs_dir / f".queue_claim_step{step}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _port_for(step: int, jobs_dir: Path) -> int:
    # A resumed job must be served on the port its config.json already points at.
    for bench in _BENCHES:
        cfg = jobs_dir / _job_name(step, bench) / "config.json"
        if cfg.exists():
            m = re.search(r":(\d+)/v1", json.loads(cfg.read_text())["agents"][0]["kwargs"]["api_base"])
            if m:
                return int(m.group(1))
    return 9000 + step


def _start_vllm(step: int, gpu: str, port: int, hf_dir: Path, mem_util: float) -> str:
    name = f"ota-vllm-sft-step{step}"
    running = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{name}$", "-f", "status=running"],
                             capture_output=True, text=True).stdout.strip()
    if running and subprocess.run(["curl", "-sf", f"localhost:{port}/v1/models"], capture_output=True).returncode == 0:
        return name  # adopt the server that is already up
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run([
        "docker", "run", "-d", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "USER=murray", "-e", "LOGNAME=murray", "-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "--ipc=host",
        "--gpus", f'"device={gpu}"', "-p", f"{port}:{port}", "-v", f"{hf_dir}:/model:ro", _VLLM_IMAGE, "/model",
        "--served-model-name", f"ota-lora-r64-step{step}", "--host", "0.0.0.0", "--port", str(port),
        "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(mem_util), "--disable-custom-all-reduce",
        "--max-model-len", "32768", "--trust-remote-code", "--enable-prefix-caching",
    ], check=True, capture_output=True)
    for _ in range(90):
        if subprocess.run(["curl", "-sf", f"localhost:{port}/v1/models"], capture_output=True).returncode == 0:
            return name
        time.sleep(10)
    raise RuntimeError(f"vLLM for step {step} did not come up on GPU {gpu}")


def _run_job(step: int, bench: str, port: int, n: int, ota_root: Path) -> subprocess.Popen:
    job_dir = ota_root / "eval_jobs" / _job_name(step, bench)
    log = open(ota_root / "logs" / f"queue_{_job_name(step, bench)}.log", "a")
    if job_dir.exists():
        cfg = json.loads((job_dir / "config.json").read_text())
        cfg["n_concurrent_trials"] = n
        (job_dir / "config.json").write_text(json.dumps(cfg, indent=4))
        cmd = [str(ota_root / "venvs/harbor/bin/harbor"), "jobs", "resume", "-p", str(job_dir)]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    env = dict(os.environ, API_BASE=f"http://localhost:{port}/v1", JOB_NAME=_job_name(step, bench))
    env.pop("AGENT_KWARGS", None)  # SFT checkpoints are served without the base model's thinking kwarg
    cmd = [str(_HERE / "run_eval.sh"), "swebench" if bench == "swebench100" else "tb2",
           f"ota-lora-r64-step{step}", "-n", str(n)]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def _worker(gpu: str, queue: list[int], lock: threading.Lock, args) -> None:
    jobs_dir = args.ota_root / "eval_jobs"
    while True:
        with lock:
            step = None
            for s in [s for s in queue if (args.run_dir / f"hf/step_{s}/.complete").exists()]:
                queue.remove(s)
                if _claim(jobs_dir, s):  # another queue process may already own this step
                    step = s
                    break
            if step is None and not queue:
                _log(f"GPU {gpu}: queue empty, done")
                return
        if step is None:
            time.sleep(300)
            continue
        if all(_finished(jobs_dir / _job_name(step, b)) for b in _BENCHES):
            _log(f"GPU {gpu}: step {step} already evaluated, skipping")
            continue
        port = _port_for(step, jobs_dir)
        _log(f"GPU {gpu}: serving step {step} on :{port}")
        name = _start_vllm(step, gpu, port, args.run_dir / f"hf/step_{step}", args.gpu_mem_util)
        # jobs that are already running (e.g. adopted after a restart of this script) are waited on, not relaunched
        procs = [_run_job(step, b, port, args.n, args.ota_root) for b in _BENCHES
                 if not _finished(jobs_dir / _job_name(step, b)) and not _active(_job_name(step, b))]
        for p in procs:
            p.wait()
        while any(_active(_job_name(step, b)) for b in _BENCHES):
            time.sleep(120)
        _log(f"GPU {gpu}: step {step} done")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--steps", required=True, help="checkpoint steps in priority order")
    parser.add_argument("--n", type=int, default=8, help="concurrent trials per job")
    parser.add_argument("--gpu-mem-util", type=float, default=0.92, help="vLLM --gpu-memory-utilization")
    parser.add_argument("--ota-root", type=Path, default=Path("/data/murray/ota_sft"))
    parser.add_argument("--run-dir", type=Path, default=Path("/data/murray/ota_sft/runs/ota_lora_r64_lr4e-4"))
    args = parser.parse_args()

    queue = [int(s) for s in args.steps.split(",")]
    lock = threading.Lock()
    threads = [threading.Thread(target=_worker, args=(g, queue, lock, args)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
