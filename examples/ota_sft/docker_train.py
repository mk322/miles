"""Start `run_qwen3_8b_ota_sft.py train` inside the Miles image on this box.

Mounts this checkout over the image's miles path, the model/dataset/run dirs under OTA_ROOT,
and forwards WANDB_API_KEY (from the environment, else ~/.netrc) so runs report to the
caller's W&B account. The container runs as the calling user, not root, so run files stay
owned by them. On sm80 (A100) use the image from `docker/Dockerfile.sm80` (the default), which
drops flash-attn-4 and makes /root writable; build it once with:

  docker build -t miles-ota:v0.1.3-sm80 -f examples/ota_sft/docker/Dockerfile.sm80 examples/ota_sft/docker

Usage:
  python examples/ota_sft/docker_train.py --name ota-lora --gpus 4,5,6,7 -- \
      --finetune-type lora --run-id ota_lora_r64
"""

import argparse
import getpass
import netrc
import os
import shlex
import subprocess
from pathlib import Path

_IMAGE = "miles-ota:v0.1.3-sm80"
_REPO = Path(__file__).resolve().parents[2]


def _wandb_key() -> str | None:
    if key := os.environ.get("WANDB_API_KEY"):
        return key
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        return None
    return auth[2] if auth else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated device ids")
    parser.add_argument("--ota-root", default="/data/murray/ota_sft")
    parser.add_argument("--image", default=_IMAGE)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_args = [a for a in args.train_args if a != "--"]
    log = f"/root/shared_data/{args.name}.log"
    inner = f"python examples/ota_sft/run_qwen3_8b_ota_sft.py train {shlex.join(train_args)} > {log} 2>&1"

    env = dict(os.environ)
    user = getpass.getuser()
    cmd = [
        "docker", "run", "-d", "--name", args.name,
        "--gpus", f'"device={args.gpus}"',
        "--ipc=host", "--shm-size", "64g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{_REPO}:/root/miles",
        "-v", f"{args.ota_root}/models:/root/models",
        "-v", f"{args.ota_root}/datasets:/root/datasets",
        "-v", f"{args.ota_root}/runs:/root/shared_data",
        "-w", "/root/miles",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={user}", "-e", f"LOGNAME={user}", "-e", "HOME=/tmp",
        # keep wandb's local files with the run instead of in the mounted checkout
        "-e", "WANDB_DIR=/root/shared_data/wandb",
    ]
    if key := _wandb_key():
        env["WANDB_API_KEY"] = key
        cmd += ["-e", "WANDB_API_KEY"]  # value comes from env, not argv
    else:
        print("WANDB_API_KEY not found; the run will not report to W&B")
    cmd += ["--entrypoint", "bash", args.image, "-c", inner]
    subprocess.run(cmd, env=env, check=True)
    print(f"started {args.name}; log: {args.ota_root}/runs/{args.name}.log")


if __name__ == "__main__":
    main()
