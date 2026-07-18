#!/usr/bin/env bash
# Vanilla OPD baseline on raw image QA parquet data.
#
# Current experiment defaults:
#   - AdamW + constant scheduler
#   - lr = 1e-6
#   - train batch size = 128
#   - one greedy student rollout per prompt
#   - 2 epochs
#   - top-k K = 32 when a top-k distillation loss is selected
#
# The student generates trajectories; the teacher only provides distillation targets.

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
# TRAIN_FILES=${TRAIN_FILES:-"['dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/train-00000-of-00006.parquet', 'dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/train-00001-of-00006.parquet', 'dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/train-00002-of-00006.parquet', 'dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/train-00003-of-00006.parquet', 'dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/train-00004-of-00006.parquet', 'dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/train-00005-of-00006.parquet']"}
TRAIN_FILES=${TRAIN_FILES:-"['dataset/offline/Qwen3-VL-2B-Thinking/PAPO_ViRL39K_train.parquet']"}
VAL_FILES=${VAL_FILES:-"['dataset/hiyouga/data/validation-00000-of-00001.parquet']"}


DATA_SOURCE=${DATA_SOURCE:-PAPOGalaxy/PAPO_ViRL39K_train}
VAL_DATA_SOURCE=${VAL_DATA_SOURCE:-hiyouga/validation}
IMAGE_KEY=${IMAGE_KEY:-images}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-200704}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-1003520}
RESPONSE_KEY=${RESPONSE_KEY:-response}
PROMPT_INSTRUCTION=${PROMPT_INSTRUCTION:-"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"<image>
{problem}

{instruction}"}
export RAW_IMAGE_QA_PROMPT_INSTRUCTION="$PROMPT_INSTRUCTION"
export RAW_IMAGE_QA_PROMPT_TEMPLATE="$PROMPT_TEMPLATE"

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
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_NUM_REPLICAS=${TEACHER_NUM_REPLICAS:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-$(( TEACHER_TP * TEACHER_NUM_REPLICAS ))}
teacher_backend=${TEACHER_BACKEND:-rollout}

# ---- paper-style hyperparameters ----
train_batch_size=${TRAIN_BATCH_SIZE:-1}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-1}
val_max_samples=${VAL_MAX_SAMPLES:-512}
rollout_n=${ROLLOUT_N:-1}
if [[ "$rollout_n" != 1 ]]; then
    echo "Greedy rollout requires ROLLOUT_N=1, got: $rollout_n" >&2
    exit 1
fi
total_epochs=${TOTAL_EPOCHS:-2}
actor_lr=${ACTOR_LR:-1e-6}
lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO:-0.0}

max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-forward_kl_topk}
use_policy_gradient=${USE_POLICY_GRADIENT:-False}
offline_response=${OFFLINE_RESPONSE:-False}
case "$offline_response" in
    True|true|TRUE|1|Yes|yes|YES) offline_response_enabled=True ;;
    False|false|FALSE|0|No|no|NO) offline_response_enabled=False ;;
    *)
        echo "OFFLINE_RESPONSE must be True or False, got: $offline_response" >&2
        exit 1
        ;;
esac
if [[ "$offline_response_enabled" == True ]]; then
    case "$use_policy_gradient" in
        False|false|FALSE|0|No|no|NO) : ;;
        *)
            echo "OFFLINE_RESPONSE=True requires USE_POLICY_GRADIENT=False." >&2
            exit 1
            ;;
    esac
    rollout_temperature=1.0
else
    rollout_temperature=0
fi
distillation_topk=${DISTILLATION_TOPK:-32}
kl_mix_alpha=${KL_MIX_ALPHA:-0.5}
use_perception_score=${USE_PERCEPTION_SCORE:-False}
if [[ -n "${TOPK_MODE:-}" ]]; then
    topk_mode=${TOPK_MODE}
else
    case "$use_perception_score" in
        True|true|TRUE|1|Yes|yes|YES) topk_mode=prob_perception ;;
        False|false|FALSE|0|No|no|NO) topk_mode=prob ;;
        *)
            echo "USE_PERCEPTION_SCORE must be True or False, got: $use_perception_score" >&2
            exit 1
            ;;
    esac
fi
if [[ -n "${PERCEPTION_CANDIDATE_TOPK:-}" ]]; then
    perception_candidate_topk=${PERCEPTION_CANDIDATE_TOPK}
elif (( distillation_topk < 64 )); then
    perception_candidate_topk=128
else
    perception_candidate_topk=$(( distillation_topk * 2 ))
fi
use_tail_bucket=${USE_TAIL_BUCKET:-null}

# ---- optional student LoRA (uses verl's native FSDP/PEFT path) ----
use_student_lora=${USE_STUDENT_LORA:-False}
student_lora_rank=${STUDENT_LORA_RANK:-64}
student_lora_alpha=${STUDENT_LORA_ALPHA:-32}
student_lora_target_modules=${STUDENT_LORA_TARGET_MODULES:-all-linear}
student_lora_exclude_modules=${STUDENT_LORA_EXCLUDE_MODULES:-'.*visual.*'}
case "$use_student_lora" in
    True|true|TRUE|1|Yes|yes|YES) student_lora_enabled=True ;;
    False|false|FALSE|0|No|no|NO) student_lora_enabled=False ;;
    *)
        echo "USE_STUDENT_LORA must be True or False, got: $use_student_lora" >&2
        exit 1
        ;;
esac
if [[ "$student_lora_enabled" == True ]]; then
    if ! [[ "$student_lora_rank" =~ ^[1-9][0-9]*$ ]]; then
        echo "STUDENT_LORA_RANK must be a positive integer, got: $student_lora_rank" >&2
        exit 1
    fi
    if ! [[ "$student_lora_alpha" =~ ^[1-9][0-9]*$ ]]; then
        echo "STUDENT_LORA_ALPHA must be a positive integer, got: $student_lora_alpha" >&2
        exit 1
    fi
fi

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.8}

save_freq=${SAVE_FREQ:-10}
if [[ "$offline_response_enabled" == True ]]; then
    test_freq=${TEST_FREQ:--1}
else
    test_freq=${TEST_FREQ:-25}
fi
val_before_train=${VAL_BEFORE_TRAIN:-False}
logger=${LOGGER:-'["console","swanlab"]'}
project_name=${PROJECT_NAME:-perception_kd}
if [[ "$offline_response_enabled" == True && "$student_lora_enabled" == True ]]; then
    default_experiment_name=${RUN_NAME}_${distillation_loss_mode}_lora_r${student_lora_rank}_perception_kd
elif [[ "$offline_response_enabled" == True ]]; then
    default_experiment_name=${RUN_NAME}_${distillation_loss_mode}_perception_kd
elif [[ "$student_lora_enabled" == True ]]; then
    default_experiment_name=${RUN_NAME}_${distillation_loss_mode}_lora_r${student_lora_rank}_greedy_n1_vanilla
else
    default_experiment_name=${RUN_NAME}_${distillation_loss_mode}_greedy_n1_vanilla
fi
experiment_name=${EXPERIMENT_NAME:-$default_experiment_name}

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
    data.image_key=${IMAGE_KEY}
    +data.answer_key=answer
    +data.default_data_source="$DATA_SOURCE"
    +data.val_default_data_source="$VAL_DATA_SOURCE"
    +data.image_min_pixels=${IMAGE_MIN_PIXELS}
    +data.image_max_pixels=${IMAGE_MAX_PIXELS}
    +data.mm_processor_kwargs.min_pixels=${IMAGE_MIN_PIXELS}
    +data.mm_processor_kwargs.max_pixels=${IMAGE_MAX_PIXELS}
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
if [[ "$offline_response_enabled" == True ]]; then
    DATA+=(+data.response_key="$RESPONSE_KEY")
fi

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.do_sample=False
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

if [[ "$student_lora_enabled" == True ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${student_lora_rank}
        actor_rollout_ref.model.lora_alpha=${student_lora_alpha}
        actor_rollout_ref.model.target_modules="${student_lora_target_modules}"
        actor_rollout_ref.model.exclude_modules="${student_lora_exclude_modules}"
    )
    if [[ "$offline_response_enabled" == False ]]; then
        ROLLOUT+=(
            actor_rollout_ref.rollout.load_format=safetensors
            actor_rollout_ref.rollout.layered_summon=True
        )
    fi
fi

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="$logger"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${val_before_train}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

REWARD=(
    reward.custom_reward_function.path="$SCRIPT_DIR/zero_reward.py"
    reward.custom_reward_function.name=compute_score
)

DISTILL=(
    distillation.enabled=True
    distillation.offline_response=${offline_response_enabled}
    distillation.teacher_backend=${teacher_backend}
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.topk_mode=${topk_mode}
    distillation.distillation_loss.perception_candidate_topk=${perception_candidate_topk}
    distillation.distillation_loss.use_tail_bucket=${use_tail_bucket}
    distillation.distillation_loss.kl_mix_alpha=${kl_mix_alpha}
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
