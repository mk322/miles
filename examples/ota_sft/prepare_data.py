"""Convert an OpenThoughts-Agent SFT parquet snapshot into the jsonl Miles reads.

Keeps exactly the rows LlamaFactory's SharegptDatasetConverter keeps: messages must
alternate user/assistant and end on an assistant turn, otherwise the row is dropped
as ``broken_data``. Row order is preserved, since OpenThoughts-Agent trains with
``disable_shuffling=True``.

Usage:
  python examples/ota_sft/prepare_data.py \
      --src $OTA_ROOT/datasets/OpenThoughts-Agent-SFT-10K \
      --dst $OTA_ROOT/datasets/ota_sft_10k.jsonl
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def _is_lf_valid(messages: list[dict]) -> bool:
    roles = [m["role"] for m in messages]
    return len(roles) % 2 == 0 and roles == ["user", "assistant"] * (len(roles) // 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(args.src.rglob("*.parquet"))
    rows = pq.read_table(shards).to_pylist()
    kept = dropped = 0
    with args.dst.open("w") as f:
        for row in rows:
            messages = [{"role": m["role"], "content": m["content"]} for m in row["conversations"]]
            if not _is_lf_valid(messages):
                dropped += 1
                continue
            f.write(json.dumps({"messages": messages, "metadata": {"task": row["task"]}}) + "\n")
            kept += 1
    print(f"{len(shards)} shards, kept {kept}, dropped {dropped} (broken role order) -> {args.dst}")


if __name__ == "__main__":
    main()
