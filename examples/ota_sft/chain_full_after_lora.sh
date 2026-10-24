#!/usr/bin/env bash
# Wait for the LoRA run to finish cleanly, then start the full-parameter run on the same GPUs.
# Usage: nohup examples/ota_sft/chain_full_after_lora.sh <lora_container> <lora_run_dir> <gpus> > chain.log 2>&1 &
set -uo pipefail
LORA=${1:?lora container}; LORA_RUN=${2:?lora run dir}; GPUS=${3:?gpus}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
note() { echo "$(date '+%F %T') $*"; }

note "waiting for $LORA to exit"
while docker ps -q -f name="^${LORA}$" -f status=running | grep -q .; do sleep 60; done
code=$(docker inspect -f '{{.State.ExitCode}}' "$LORA")
last=$(ls -d "$LORA_RUN"/hf/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1)
note "$LORA exited with code $code; last HF export step_${last}"
if [ "$code" != 0 ] || [ ! -f "$LORA_RUN/hf/step_${last}/.complete" ] || [ "$last" -lt 664 ]; then
  note "LoRA run did not finish cleanly; not starting the full-parameter run"; exit 1
fi

note "starting full-parameter SFT on GPUs $GPUS"
python3 "$HERE/docker_train.py" --name ota_full --gpus "$GPUS" -- \
  --finetune-type full --run-id ota_full_lr4e-5
note "launched"
