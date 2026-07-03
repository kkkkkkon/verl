#!/usr/bin/env bash
# Vanilla OPD baseline on raw image QA parquet data.
#
# This runner follows the DOPD paper's VLM implementation settings:
#   - AdamW + cosine scheduler
#   - lr = 5e-6
#   - train batch size = 64
#   - rollout samples = 4
#   - max training steps = 300
#   - top-k K = 128 when a top-k distillation loss is selected
#
# It intentionally does not implement DOPD's dual/advantage-aware routing.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- model pair, usually set by the small wrapper scripts ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
RUN_NAME=${RUN_NAME:-qwen3vl_8b_to_2b}

# ---- data ----
DATA_DIR=${DATA_DIR:-$HOME/data/image_qa}
TRAIN_FILE=${TRAIN_FILE:-}
VAL_FILE=${VAL_FILE:-}
TRAIN_FILES=${TRAIN_FILES:-}
VAL_FILES=${VAL_FILES:-}
DATA_SOURCE=${DATA_SOURCE:-PAPOGalaxy/PAPO_ViRL39K_train}
PROMPT_INSTRUCTION=${PROMPT_INSTRUCTION:-"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"<image>
{problem}

{instruction}"}

build_file_list() {
    local result="["
    local separator=""
    local file

    for file in "$@"; do
        result="${result}${separator}'${file}'"
        separator=", "
    done

    result="${result}]"
    printf '%s\n' "$result"
}

# ---- resources ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_TP=${TEACHER_TP:-4}
TEACHER_NUM_REPLICAS=${TEACHER_NUM_REPLICAS:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-$(( TEACHER_TP * TEACHER_NUM_REPLICAS ))}

# ---- paper-style hyperparameters ----
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
val_max_samples=${VAL_MAX_SAMPLES:-512}
rollout_n=${ROLLOUT_N:-4}
total_training_steps=${TOTAL_TRAINING_STEPS:-300}
total_epochs=${TOTAL_EPOCHS:-30}
actor_lr=${ACTOR_LR:-5e-6}
lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO:-0.0}

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-128}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:--1}
logger=${LOGGER:-'["console","wandb"]'}
project_name=${PROJECT_NAME:-vanilla_opd}
experiment_name=${EXPERIMENT_NAME:-${RUN_NAME}_${distillation_loss_mode}_vanilla}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

train_file_paths=()
if [[ -z "$TRAIN_FILES" ]]; then
    if [[ -n "$TRAIN_FILE" ]]; then
        train_file_paths=("$TRAIN_FILE")
    else
        train_shards=("$DATA_DIR"/train-*.parquet)
        if [[ -f "${train_shards[0]}" ]]; then
            train_file_paths=("${train_shards[@]}")
        else
            train_file_paths=("$DATA_DIR/train.parquet")
        fi
    fi

    TRAIN_FILES=$(build_file_list "${train_file_paths[@]}")
fi

if [[ ${#train_file_paths[@]} -gt 0 ]]; then
    for file in "${train_file_paths[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Missing training file: $file" >&2
            echo "Set DATA_DIR to the shard directory, TRAIN_FILE to one parquet, or TRAIN_FILES to a Hydra list." >&2
            exit 1
        fi
    done
fi

val_file_paths=()
if [[ -z "$VAL_FILES" ]]; then
    if [[ -n "$VAL_FILE" ]]; then
        val_file_paths=("$VAL_FILE")
        VAL_FILES=$(build_file_list "${val_file_paths[@]}")
    else
        VAL_FILES="$TRAIN_FILES"
    fi
fi

if [[ ${#val_file_paths[@]} -gt 0 ]]; then
    for file in "${val_file_paths[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "VAL_FILE=$file not found; using training files as validation placeholder." >&2
            VAL_FILES="$TRAIN_FILES"
            break
        fi
    done
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.prompt_key=problem
    data.image_key=image
    +data.answer_key=answer
    +data.default_data_source="$DATA_SOURCE"
    +data.prompt_instruction="$PROMPT_INSTRUCTION"
    +data.prompt_template="$PROMPT_TEMPLATE"
    data.custom_cls.path="$SCRIPT_DIR/raw_image_qa_dataset.py"
    data.custom_cls.name=RawImageQADataset
    data.train_batch_size=${train_batch_size}
    data.val_batch_size=${train_batch_size}
    data.val_max_samples=${val_max_samples}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.return_multi_modal_inputs=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="$logger"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

REWARD=(
    reward.custom_reward_function.path="$SCRIPT_DIR/zero_reward.py"
    reward.custom_reward_function.name=compute_score
)

DISTILL=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${DISTILL[@]}" \
    "$@"
