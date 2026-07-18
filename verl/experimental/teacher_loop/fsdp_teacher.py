# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import partial

import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation.topk_support import (
    TOPK_MODE_PROB_PERCEPTION,
    build_prob_perception_support_from_topk_logprobs,
    build_topk_logprobs_from_logits,
)
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker


def _teacher_topk_processor(
    *,
    candidate_topk: int,
    chunk_size: int,
    student_logits: torch.Tensor | None = None,
    model_output: dict | None = None,
    data: TensorDict | None = None,
    dp_group=None,
    **_,
):
    """Dual-use FSDP logits processor/final loss callback."""
    if student_logits is not None:
        teacher_ids, teacher_logprobs = build_topk_logprobs_from_logits(
            student_logits,
            topk=candidate_topk,
            chunk_size=chunk_size,
        )
        return {
            "teacher_ids": teacher_ids,
            "teacher_logprobs": teacher_logprobs,
        }

    reference = next(iter(model_output.values()))
    device = reference.values().device if reference.is_nested else reference.device
    return torch.zeros((), dtype=torch.float32, device=device), {}


def build_offline_fsdp_teacher_worker_config(
    config: DictConfig,
    distillation_config: DistillationConfig,
) -> TrainingWorkerConfig:
    """Build a forward-only FSDP worker config for the single offline teacher."""
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    if actor_config.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError(
            "distillation.teacher_backend='fsdp' requires the actor FSDP engine configuration, "
            f"got actor strategy {actor_config.strategy!r}."
        )

    teacher_model_config = next(iter(distillation_config.teacher_models.values()))
    student_model_config = config.actor_rollout_ref.model
    model_config = HFModelConfig(
        path=teacher_model_config.model_path,
        load_tokenizer=False,
        use_shm=student_model_config.get("use_shm", False),
        trust_remote_code=student_model_config.get("trust_remote_code", False),
        external_lib=student_model_config.get("external_lib", None),
        enable_gradient_checkpointing=False,
        use_remove_padding=student_model_config.get("use_remove_padding", True),
        use_fused_kernels=False,
    )

    # BaseConfig instances are frozen after construction, so create the
    # forward-only variant atomically instead of mutating the actor engine.
    engine_config = replace(
        deepcopy(actor_config.engine),
        forward_only=True,
        forward_only_param_offload=False,
        param_offload=False,
        optimizer_offload=False,
        offload_policy=False,
        reshard_after_forward=False,
        use_fused_kernels=False,
        use_remove_padding=model_config.use_remove_padding,
        model_dtype=actor_config.engine.dtype,
        use_dynamic_bsz=True,
        infer_micro_batch_size_per_gpu=None,
        infer_max_token_len_per_gpu=(
            teacher_model_config.inference.max_num_batched_tokens or teacher_model_config.inference.max_model_len
        ),
    )
    if engine_config.infer_max_token_len_per_gpu is None:
        raise ValueError(
            "The offline FSDP teacher requires inference.max_num_batched_tokens "
            "or inference.max_model_len to configure dynamic batching."
        )

    return TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=None,
        checkpoint_config=deepcopy(actor_config.checkpoint),
    )


class OfflineFSDPTeacherWorker(TrainingWorker):
    """Forward-only FSDP teacher that builds compact KD support on GPU."""

    def __init__(
        self,
        config: TrainingWorkerConfig,
        distillation_config: DistillationConfig,
        topk_chunk_size: int = 256,
    ):
        super().__init__(config=config)
        self.distillation_config = omega_conf_to_dataclass(
            distillation_config,
            dataclass_type=DistillationConfig,
        )
        self.loss_config = self.distillation_config.distillation_loss
        self.topk_chunk_size = topk_chunk_size

    def _prepare_forward_data(self, data: TensorDict) -> None:
        defaults = {
            "calculate_entropy": False,
            "distillation_use_topk": True,
            "distillation_only": True,
            "use_fused_kernels": False,
            "use_remove_padding": self.model_config.get("use_remove_padding", True),
            "use_dynamic_bsz": self.engine_config.use_dynamic_bsz,
            "max_token_len_per_gpu": self.engine_config.infer_max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.engine_config.infer_micro_batch_size_per_gpu,
        }
        for key, value in defaults.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: value})
        # Prompt log-probabilities are untempered model probabilities. Do not
        # inherit a sampling temperature from the student rollout metadata.
        tu.assign_non_tensor(data, temperature=1.0)

    def _infer_candidates(self, data: TensorDict, candidate_topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        processor = partial(
            _teacher_topk_processor,
            candidate_topk=candidate_topk,
            chunk_size=self.topk_chunk_size,
        )
        output = self.engine.infer_batch(data, loss_function=processor)
        model_output = output["model_output"]
        return model_output["teacher_ids"], model_output["teacher_logprobs"]

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def compute_teacher_support(self, data: TensorDict) -> TensorDict:
        """Run image/masked-image teacher forwards and select support on GPU."""
        self._prepare_forward_data(data)
        candidate_topk = self.loss_config.teacher_logprob_topk

        with self.engine.eval_mode(disable_auto_offload=True):
            image_ids, image_logprobs = self._infer_candidates(data, candidate_topk)

            if self.loss_config.topk_mode == TOPK_MODE_PROB_PERCEPTION:
                if "masked_multi_modal_inputs" not in data:
                    raise KeyError("Offline perception FSDP teacher requires 'masked_multi_modal_inputs' in the batch.")
                masked_data = data.clone(recurse=False)
                masked_data["multi_modal_inputs"] = data["masked_multi_modal_inputs"]
                masked_ids, masked_logprobs = self._infer_candidates(masked_data, candidate_topk)
                if not torch.equal(image_ids.offsets(), masked_ids.offsets()):
                    raise ValueError("Image and masked-image teacher outputs have different sequence offsets.")

                support_ids, support_logprobs = build_prob_perception_support_from_topk_logprobs(
                    teacher_ids_img=image_ids.values(),
                    teacher_logprobs_img=image_logprobs.values(),
                    teacher_ids_mask=masked_ids.values(),
                    teacher_logprobs_mask=masked_logprobs.values(),
                    topk=self.loss_config.topk,
                    perception_candidate_topk=self.loss_config.perception_candidate_topk,
                )
                offsets = image_ids.offsets()
                teacher_ids = torch.nested.nested_tensor_from_jagged(support_ids, offsets=offsets)
                teacher_logprobs = torch.nested.nested_tensor_from_jagged(support_logprobs, offsets=offsets)
            else:
                teacher_ids = image_ids
                teacher_logprobs = image_logprobs

        return TensorDict(
            {
                "teacher_ids": teacher_ids.cpu(),
                "teacher_logprobs": teacher_logprobs.cpu(),
            },
            batch_size=data.batch_size,
        )
