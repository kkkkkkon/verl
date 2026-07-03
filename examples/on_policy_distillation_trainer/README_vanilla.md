# Raw Image QA Vanilla OPD Baselines

These scripts run five teacher/student scale comparisons as vanilla on-policy
distillation baselines on raw image QA parquet data.

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

The shared runner follows the DOPD paper's VLM implementation settings:
`lr=5e-6`, `train_batch_size=64`, `rollout.n=4`,
`trainer.total_training_steps=300`, AdamW, cosine LR scheduler, and `K=128`
when a top-k distillation loss is selected. It does not implement DOPD routing,
privileged input, or advantage-aware weighting.

## Data

The raw parquet rows are `image/problem/answer`. There is no offline
preprocessing step. The training scripts use `data.custom_cls` to load
`raw_image_qa_dataset.py`, a thin runtime adapter on top of verl's existing
`RLHFDataset`.

Set `DATA_DIR` to a directory containing `train-*.parquet` shards, or set
`TRAIN_FILES` directly to a Hydra list:

```bash
DATA_DIR=/path/to/dataset_dir \
bash verl/examples/on_policy_distillation_trainer/run_qwen3vl_8b_to_2b_fsdp.sh
```

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

The student rollout and teacher-logprob pass then share the same tokenized
`raw_prompt` and the same image payload through verl's native distillation path,
so the distillation template stays consistent.

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
DATA_DIR=/path/to/dataset_dir \
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
