# Raw Image QA Vanilla OPD Baselines

These scripts run five teacher/student scale comparisons as vanilla on-policy
distillation baselines on raw image QA parquet data. Each prompt receives one
greedy student-generated trajectory; the teacher only supplies distillation
targets.

The default dataset source is `PAPOGalaxy/PAPO_ViRL39K_train`. Because this is
an image-text dataset, the default wrappers use Qwen3-VL scale analogues rather
than the text-only Qwen3 pairs from the paper:

| Reference text-only pair | Default VLM pair | Script |
| --- | --- | --- |
| Qwen3-8B -> Qwen3-0.6B | Qwen3-VL-32B -> Qwen3-VL-2B | `run_qwen3vl_32b_to_2b_fsdp.sh` |
| Qwen3-8B -> Qwen3-1.7B | Qwen3-VL-32B -> Qwen3-VL-4B | `run_qwen3vl_32b_to_4b_fsdp.sh` |
| Qwen3-4B -> Qwen3-0.6B | Qwen3-VL-8B -> Qwen3-VL-2B | `run_qwen3vl_8b_to_2b_fsdp.sh` |
| Qwen3-4B -> Qwen3-1.7B | Qwen3-VL-8B -> Qwen3-VL-4B | `run_qwen3vl_8b_to_4b_fsdp.sh` |
| Qwen3-1.7B -> Qwen3-0.6B | Qwen3-VL-4B -> Qwen3-VL-2B | `run_qwen3vl_4b_to_2b_fsdp.sh` |

The shared runner keeps the DOPD-style optimization settings:
`lr=5e-6`, `train_batch_size=64`, `rollout.n=1`, student temperature `0`,
`trainer.total_training_steps=300`, AdamW, cosine LR scheduler, and `K=128`
when a top-k distillation loss is selected. It does not implement DOPD routing,
privileged input, or advantage-aware weighting.

## Data

The raw parquet rows are `image/problem/answer`. There is no offline
preprocessing step. The training scripts use `data.custom_cls` to load
`raw_image_qa_dataset.py`, a thin runtime adapter on top of verl's existing
`RLHFDataset`. It wraps the `problem` string into a user chat message, injects
the original `image` payload, and maps `answer` to `reward_model.ground_truth`.

`TRAIN_FILES` defaults to the six local shards under
`dataset/PAPOGalaxy/PAPO_ViRL39K_train/data/`. You can still override it with a
Hydra list:

```bash
TRAIN_FILES="['/path/train-00000-of-00006.parquet', '/path/train-00001-of-00006.parquet']" \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
```

`VAL_FILES` defaults to `TRAIN_FILES`, and validation is disabled by default
(`trainer.val_before_train=False`, `trainer.test_freq=-1`).

The default prompt template is:

```text
<image>
{problem}

{instruction}
```

`PROMPT_TEMPLATE` and `PROMPT_INSTRUCTION` are passed through environment
variables instead of Hydra overrides, so tags such as `<think>`, `<answer>`, and
`<image>` are safe.

## Run One Baseline

```bash
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
```

## Run All Five

```bash
bash verl/examples/on_policy_distillation_trainer/run_all_vanilla_opd_fsdp.sh
```

Useful overrides:

```bash
TRAIN_FILES="['/path/train-00000-of-00006.parquet', '/path/train-00001-of-00006.parquet']" \
LOGGER='["console"]' \
DISTILLATION_LOSS_MODE=k1 \
USE_POLICY_GRADIENT=True \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
```

To change the shared distillation prompt template:

```bash
PROMPT_TEMPLATE='<image>
{problem}

{instruction}' \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
```

For top-k supervised distillation instead of sampled-token vanilla OPD:

```bash
DISTILLATION_LOSS_MODE=forward_kl_topk \
USE_POLICY_GRADIENT=False \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
```

For perception top-k with a 32-token support, normal-image probability top-128
candidate pool, and a 50/50 forward/reverse KL mixture:

```bash
USE_PERCEPTION_SCORE=True \
DISTILLATION_LOSS_MODE=mixed_kl \
KL_MIX_ALPHA=0.5 \
DISTILLATION_TOPK=32 \
PERCEPTION_CANDIDATE_TOPK=128 \
USE_POLICY_GRADIENT=False \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_4b_to_2b_fsdp.sh
```

Perception mode selects 16 probability tokens and 16 unique perception tokens.
The perception group is restricted to the normal-image teacher probability
top-128. A tail bucket is enabled automatically for perception and reverse-KL
runs, as well as mixed-KL runs; set `USE_TAIL_BUCKET=False` only for a
support-conditioned ablation.
When `USE_PERCEPTION_SCORE` is omitted or false, the runner uses the original
teacher-probability top-k and skips the masked-image teacher forward. An
explicit `TOPK_MODE` remains supported and takes precedence over the boolean
switch.

For `mixed_kl`, `KL_MIX_ALPHA` is the forward-KL weight:
`alpha * KL(teacher || student) + (1 - alpha) * KL(student || teacher)`.
`forward_reverse_kl` is accepted as an alias, and `forward_kl` aliases
`forward_kl_topk`.

The logged `distillation/perception_teacher_mass`,
`distillation/perception_teacher_mean_probability`, and
`distillation/perception_teacher_mass_fraction` metrics show whether the added
perception tokens carry enough teacher probability to affect training. Mixed
runs additionally log `distillation/forward_kl_component` and
`distillation/reverse_kl_component`.

## Greedy Student Rollout

This branch generates one deterministic trajectory per prompt with the student.
After each optimizer step, verl synchronizes the updated student weights to its
rollout replica through the normal synchronous trainer path.

```bash
ROLLOUT_N=1 \
USE_STUDENT_LORA=False \
USE_PERCEPTION_SCORE=True \
DISTILLATION_LOSS_MODE=mixed_kl \
KL_MIX_ALPHA=0.5 \
DISTILLATION_TOPK=32 \
PERCEPTION_CANDIDATE_TOPK=128 \
USE_POLICY_GRADIENT=False \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_4b_to_2b_fsdp.sh
```

The runner fixes rollout `temperature=0`, `top_p=1`, `top_k=-1`,
`do_sample=False`, and requires `ROLLOUT_N=1`. The teacher is used only for
prompt-logprob scoring over the student-generated sequence.

## Student LoRA

The runner retains verl's native FSDP/PEFT LoRA path, but LoRA is disabled by
default. Enable it explicitly with `USE_STUDENT_LORA=True`; the visual tower is
excluded by default to reduce memory use:

```bash
USE_STUDENT_LORA=True \
STUDENT_LORA_RANK=64 \
STUDENT_LORA_ALPHA=32 \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_4b_to_2b_fsdp.sh
```

This adds `actor_rollout_ref.model.lora_rank`, `lora_alpha`, `target_modules`,
and `exclude_modules`, plus the rollout-side `load_format=safetensors` and
`layered_summon=True` settings. The teacher remains unchanged.
