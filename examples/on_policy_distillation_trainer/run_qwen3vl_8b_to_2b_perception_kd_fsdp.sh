#!/usr/bin/env bash
# Qwen3-VL 8B teacher to Qwen3-VL 2B student with offline perception KD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=3,4,5,6

export RUN_NAME=${RUN_NAME:-qwen3vl_8b_to_2b_thinking_vanila}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Thinking}
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Thinking}

export TEACHER_TP=${TEACHER_TP:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export NNODES=${NNODES:-1}
export SAVE_FREQ=${SAVE_FREQ:-10}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}

# Paper training setup: per-device batch size 1 with 512 gradient
# accumulation steps. verl's ppo_mini_batch_size is global across the
# student data-parallel workers, so the corresponding global batch is
# micro_batch_size_per_gpu * accumulation_steps * student_world_size.
micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-512}
student_world_size=$((NNODES * NGPUS_PER_NODE))
global_optimizer_batch_size=$((micro_batch_size_per_gpu * gradient_accumulation_steps * student_world_size))

exec "$SCRIPT_DIR/run_perception_kd_fsdp.sh" \
    data.train_batch_size=${global_optimizer_batch_size} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${global_optimizer_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.total_epochs=2 \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    "$@"
