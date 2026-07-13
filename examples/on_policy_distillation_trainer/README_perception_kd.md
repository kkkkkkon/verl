# Offline Perception KD

This branch supports direct distillation from multimodal responses stored in
the dataset. It does not sample a student rollout and does not create a student
vLLM server. Student GPUs contain only the trainable FSDP actor; teacher GPUs
still run the teacher vLLM server used to produce distillation targets.

## Dataset schema

Each parquet row must contain:

- `image` or `images`: one raw image or a list of raw images. Paths are also
  accepted by the existing adapter, but are not required.
- `problem`: the user question.
- `answer`: the ground truth retained for metadata and later evaluation.
- `response`: the offline assistant response used for distillation.

The adapter builds one user chat message containing all images and the rendered
problem. It then appends `response` as the assistant message and applies the
student model's chat template. The response suffix receives the same
`response_mask` used by online distillation, so only assistant response tokens
contribute to the loss.

## Run 4B teacher to 2B student

```bash
TRAIN_FILES="['/path/to/train-00000-of-00006.parquet', '/path/to/train-00001-of-00006.parquet']" \
VAL_FILES="['/path/to/train-00000-of-00006.parquet']" \
NGPUS_PER_NODE=2 \
TEACHER_TP=2 \
TEACHER_WORLD_SIZE=2 \
bash examples/on_policy_distillation_trainer/run_qwen3vl_4b_to_2b_perception_kd_fsdp.sh
```

`NGPUS_PER_NODE` is the number of student FSDP actor GPUs.
`TEACHER_WORLD_SIZE` is the size of the separate teacher resource pool. The
example above therefore requests four GPUs in total: two student actor GPUs and
two teacher vLLM GPUs.

The launcher defaults to:

```text
OFFLINE_RESPONSE=True
USE_POLICY_GRADIENT=False
USE_PERCEPTION_SCORE=True
DISTILLATION_LOSS_MODE=mixed_kl
DISTILLATION_TOPK=32
PERCEPTION_CANDIDATE_TOPK=128
KL_MIX_ALPHA=0.5
USE_STUDENT_LORA=False
```

Generated-response validation requires a student rollout server and is
therefore disabled in this mode. Keep `trainer.val_before_train=False` and
`TEST_FREQ=-1`; evaluate saved checkpoints with a separate inference job.

Although the shared launcher still supplies `actor_rollout_ref.rollout.*`
values needed by common model and batching configuration, the offline trainer
uses `Role.Actor`. It skips both `LLMServerManager.create(...)` and rollout
checkpoint synchronization, so those values do not instantiate or reserve
memory for a student vLLM process.
