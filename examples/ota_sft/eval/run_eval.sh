#!/usr/bin/env bash
# Run one OpenThoughts-Agent-style Harbor eval (terminus-2, local Docker) against an
# OpenAI-compatible endpoint, mirroring OpenThoughts-Agent/eval/jupiter/eval_harbor.sbatch.
#
# Usage: run_eval.sh <swebench|tb2> <served_model_name> [extra harbor args...]
# Env:   OTA_ROOT (default /data/murray/ota_sft), API_BASE (default http://localhost:8000/v1),
#        AGENT_KWARGS (newline-separated extra --agent-kwarg values), JOB_NAME.
set -euo pipefail

BENCH="${1:?bench: swebench|tb2}"
MODEL="${2:?served model name}"
shift 2

OTA_ROOT="${OTA_ROOT:-/data/murray/ota_sft}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Presets follow OpenThoughts-Agent/eval/presets/{swebench,tb2}.yaml.
case "$BENCH" in
  swebench) DATASET="$OTA_ROOT/datasets/swebench-verified-random-100-folders"; PARSER="xml" ;;
  tb2)      DATASET="$OTA_ROOT/datasets/terminal_bench_2"; PARSER="" ;;
  *) echo "unknown bench $BENCH" >&2; exit 1 ;;
esac

JOB_NAME="${JOB_NAME:-$(echo "$MODEL" | tr '/' '_')_${BENCH}_$(date +%Y%m%d_%H%M%S)}"
JOBS_DIR="$OTA_ROOT/eval_jobs"
mkdir -p "$JOBS_DIR"

# The sbatch injects model_name = hosted_vllm/<model> into a runtime copy of the yaml.
RUNTIME_CFG="$JOBS_DIR/$JOB_NAME.config.yaml"
sed "s#model_name: hosted_vllm/.*#model_name: hosted_vllm/${MODEL}#; s#metrics_endpoint: .*#metrics_endpoint: ${API_BASE%/v1}/metrics#" \
  "$HERE/terminus2_local_docker.yaml" > "$RUNTIME_CFG"

CMD=(
  "$OTA_ROOT/venvs/harbor/bin/harbor" jobs start
  -p "$DATASET"
  --config "$RUNTIME_CFG"
  --agent-kwarg "api_base=${API_BASE}"
  --agent-kwarg "key=fake_key"
  --job-name "$JOB_NAME"
  --jobs-dir "$JOBS_DIR"
  --export-traces
  --yes
)
[ -n "$PARSER" ] && CMD+=(--agent-kwarg "parser=$PARSER")
if [ -n "${AGENT_KWARGS:-}" ]; then
  while IFS= read -r kw; do [ -n "$kw" ] && CMD+=(--agent-kwarg "$kw"); done <<< "$AGENT_KWARGS"
fi

echo "[run_eval] ${CMD[*]} $*"
exec sg docker -c "$(printf '%q ' "${CMD[@]}" "$@")"
