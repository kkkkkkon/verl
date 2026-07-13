#!/usr/bin/env bash
# Qwen3-VL 8B teacher to Qwen3-VL 2B student.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME=${RUN_NAME:-qwen3vl_8b_to_2b}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Thinking}
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Thinking}
export TEACHER_TP=${TEACHER_TP:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export CUDA_VISIBLE_DEVICES=4,5,6,7

exec "$SCRIPT_DIR/run_vanilla_opd_fsdp.sh" "$@"
