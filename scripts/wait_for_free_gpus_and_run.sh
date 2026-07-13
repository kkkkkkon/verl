#!/usr/bin/env bash

set -euo pipefail

# Wait until enough GPUs look idle, then launch the Qwen3-VL distillation job.
# Defaults can be overridden from the environment:
#   REQUIRED_GPUS=2
#   CHECK_INTERVAL_SECONDS=60
#   MAX_MEMORY_USED_MB=1024
#   MAX_GPU_UTILIZATION=5
#   CANDIDATE_GPUS=6,7

REQUIRED_GPUS="${REQUIRED_GPUS:-2}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-2048}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_COMMAND=(
  bash
  examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
)
COMMAND=("${DEFAULT_COMMAND[@]}")

usage() {
  cat <<'USAGE'
Usage:
  scripts/wait_for_free_gpus_and_run.sh [trainer args...]
  scripts/wait_for_free_gpus_and_run.sh -- <command...>

Environment:
  REQUIRED_GPUS              Number of idle GPUs required. Default: 2
  CHECK_INTERVAL_SECONDS     Poll interval in seconds. Default: 60
  MAX_MEMORY_USED_MB         A GPU is idle when used memory is <= this. Default: 1024
  MAX_GPU_UTILIZATION        A GPU is idle when utilization is <= this. Default: 5
  CANDIDATE_GPUS             Optional comma-separated physical GPU IDs to monitor.

Examples:
  scripts/wait_for_free_gpus_and_run.sh
  CHECK_INTERVAL_SECONDS=30 CANDIDATE_GPUS=4,5,6,7 scripts/wait_for_free_gpus_and_run.sh
  MAX_MEMORY_USED_MB=500 scripts/wait_for_free_gpus_and_run.sh -- bash my_train.sh
USAGE
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

trim_spaces() {
  local value="$1"
  value="${value//[[:space:]]/}"
  printf '%s' "$value"
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_candidate_gpu() {
  local gpu_index="$1"
  local candidate

  [[ -z "$CANDIDATE_GPUS" ]] && return 0

  IFS=',' read -r -a candidates <<<"$CANDIDATE_GPUS"
  for candidate in "${candidates[@]}"; do
    candidate="$(trim_spaces "$candidate")"
    [[ "$gpu_index" == "$candidate" ]] && return 0
  done

  return 1
}

select_idle_gpus() {
  local gpu_index
  local memory_used
  local gpu_utilization
  local -a idle_gpus=()

  while IFS=',' read -r gpu_index memory_used gpu_utilization; do
    gpu_index="$(trim_spaces "$gpu_index")"
    memory_used="$(trim_spaces "$memory_used")"
    gpu_utilization="$(trim_spaces "$gpu_utilization")"

    is_candidate_gpu "$gpu_index" || continue
    is_nonnegative_int "$memory_used" || continue
    is_nonnegative_int "$gpu_utilization" || continue

    if (( memory_used <= MAX_MEMORY_USED_MB && gpu_utilization <= MAX_GPU_UTILIZATION )); then
      idle_gpus+=("$gpu_index")
    fi

    if (( ${#idle_gpus[@]} >= REQUIRED_GPUS )); then
      printf '%s\n' "${idle_gpus[@]:0:REQUIRED_GPUS}"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

  return 1
}

join_by_comma() {
  local joined=""
  local value

  for value in "$@"; do
    if [[ -z "$joined" ]]; then
      joined="$value"
    else
      joined="$joined,$value"
    fi
  done

  printf '%s' "$joined"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        COMMAND=("$@")
      fi
      break
      ;;
    *)
      COMMAND+=("$1")
      shift
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot monitor GPUs." >&2
  exit 1
fi

for value_name in REQUIRED_GPUS CHECK_INTERVAL_SECONDS MAX_MEMORY_USED_MB MAX_GPU_UTILIZATION; do
  value="${!value_name}"
  if ! is_nonnegative_int "$value"; then
    echo "$value_name must be an integer, got: $value" >&2
    exit 1
  fi
done

if ! is_positive_int "$REQUIRED_GPUS"; then
  echo "REQUIRED_GPUS must be greater than 0, got: $REQUIRED_GPUS" >&2
  exit 1
fi

cd "$REPO_ROOT"

log "Waiting for $REQUIRED_GPUS idle GPU(s). memory<=${MAX_MEMORY_USED_MB}MiB utilization<=${MAX_GPU_UTILIZATION}% interval=${CHECK_INTERVAL_SECONDS}s"
if [[ -n "$CANDIDATE_GPUS" ]]; then
  log "Monitoring only candidate GPU(s): $CANDIDATE_GPUS"
fi
log "Command: ${COMMAND[*]}"

while true; do
  mapfile -t selected_gpus < <(select_idle_gpus)

  if (( ${#selected_gpus[@]} >= REQUIRED_GPUS )); then
    cuda_visible_devices="$(join_by_comma "${selected_gpus[@]}")"
    log "Selected idle GPU(s): $cuda_visible_devices"
    log "Launching with CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
    exec env CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "${COMMAND[@]}"
  fi

  log "Not enough idle GPUs yet; checking again in ${CHECK_INTERVAL_SECONDS}s"
  sleep "$CHECK_INTERVAL_SECONDS"
done
