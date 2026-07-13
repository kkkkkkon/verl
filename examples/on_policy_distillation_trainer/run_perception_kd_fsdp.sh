#!/usr/bin/env bash
# Offline perception KD from dataset-provided multimodal responses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OFFLINE_RESPONSE=True
export USE_POLICY_GRADIENT=False
export USE_PERCEPTION_SCORE=${USE_PERCEPTION_SCORE:-True}
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-mixed_kl}
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
export PERCEPTION_CANDIDATE_TOPK=${PERCEPTION_CANDIDATE_TOPK:-128}
export KL_MIX_ALPHA=${KL_MIX_ALPHA:-0.5}
export USE_STUDENT_LORA=${USE_STUDENT_LORA:-False}
export PROJECT_NAME=${PROJECT_NAME:-perception_kd}

exec "$SCRIPT_DIR/run_vanilla_opd_fsdp.sh" "$@"
