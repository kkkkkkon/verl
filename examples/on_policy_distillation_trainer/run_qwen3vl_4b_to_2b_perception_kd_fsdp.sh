#!/usr/bin/env bash
# Qwen3-VL 4B teacher to Qwen3-VL 2B student with offline perception KD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME=${RUN_NAME:-qwen3vl_4b_to_2b}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
export TEACHER_TP=${TEACHER_TP:-2}

exec "$SCRIPT_DIR/run_perception_kd_fsdp.sh" "$@"
