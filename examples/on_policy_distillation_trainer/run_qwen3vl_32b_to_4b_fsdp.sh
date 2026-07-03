#!/usr/bin/env bash
# Qwen3-VL 32B teacher to Qwen3-VL 4B student.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME=${RUN_NAME:-qwen3vl_32b_to_4b}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-32B-Instruct}
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
export TEACHER_TP=${TEACHER_TP:-4}
export ROLLOUT_TP=${ROLLOUT_TP:-2}

exec "$SCRIPT_DIR/run_vanilla_opd_fsdp.sh" "$@"
