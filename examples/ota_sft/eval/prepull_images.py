"""Pre-pull every base image a Harbor dataset needs, with retries.

Harbor builds each SWE-bench task from a Dockerfile whose `FROM` is a Docker Hub image, and
Terminal-Bench tasks name a prebuilt `docker_image`. Pulling them on demand from 64 concurrent
trials trips Docker Hub's anonymous rate limit (HTTP 429), which Harbor records as an errored
trial (reward 0). Pull them once, ahead of the eval, so trials only hit the local cache.

Usage:
  python examples/ota_sft/eval/prepull_images.py /data/murray/ota_sft/datasets/swebench-verified-random-100-folders \
      /data/murray/ota_sft/datasets/terminal_bench_2 --jobs 8
"""

import argparse
import re
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_FROM = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)", re.IGNORECASE | re.MULTILINE)


def _task_images(task_dir: Path) -> set[str]:
    images = set()
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        image = tomllib.loads(toml_path.read_text()).get("environment", {}).get("docker_image")
        if image:
            images.add(image)
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        images.update(m for m in _FROM.findall(dockerfile.read_text()) if "$" not in m)
    return images


def _have(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def _pull(image: str, attempts: int) -> tuple[str, bool]:
    if _have(image):
        return image, True
    for i in range(attempts):
        result = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True)
        if result.returncode == 0:
            return image, True
        wait = min(600, 30 * 2**i)
        print(f"[retry {i + 1}/{attempts} in {wait}s] {image}: {result.stderr.strip()[-200:]}", flush=True)
        time.sleep(wait)
    return image, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=6)
    args = parser.parse_args()

    images = sorted({img for ds in args.datasets for task in ds.iterdir() if task.is_dir() for img in _task_images(task)})
    print(f"{len(images)} unique base images", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda img: _pull(img, args.attempts), images))
    failed = [img for img, ok in results if not ok]
    print(f"done: {len(images) - len(failed)} ok, {len(failed)} failed", flush=True)
    for img in failed:
        print(f"FAILED {img}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
