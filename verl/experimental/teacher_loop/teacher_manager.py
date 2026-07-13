# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from PIL import Image
from torch.nn import functional as F

from verl.trainer.distillation.topk_support import (
    TOPK_MODE_PROB_PERCEPTION,
    build_prob_perception_support_from_topk_logprobs,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    num_logprobs = (
        distillation_loss_config.teacher_logprob_topk if distillation_loss_config.loss_settings.use_topk else 0
    )
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
    }


def _mask_image_payload(image: Any) -> Any:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return Image.new(image.mode, image.size)
    if isinstance(image, bytes):
        source = Image.open(BytesIO(image))
        return Image.new(source.mode, source.size)
    if isinstance(image, str | os.PathLike):
        source = Image.open(os.fspath(image))
        return Image.new(source.mode, source.size)
    if torch.is_tensor(image):
        return torch.zeros_like(image)
    if isinstance(image, dict):
        payload = dict(image)
        if payload.get("image") is not None:
            payload["image"] = _mask_image_payload(payload["image"])
        elif payload.get("bytes") is not None:
            payload["image"] = _mask_image_payload(payload["bytes"])
            payload.pop("bytes", None)
        elif payload.get("path") is not None:
            payload["image"] = _mask_image_payload(payload["path"])
            payload.pop("path", None)
        return payload
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            return np.zeros_like(image)
    except ImportError:
        pass
    raise TypeError(f"Unsupported image payload for prob_perception masking: {type(image)}")


def _mask_multi_modal_images(multi_modal_data: dict[str, Any]) -> dict[str, Any]:
    masked_data = dict(multi_modal_data)
    images = masked_data.get("images")
    if images is None:
        return masked_data
    if isinstance(images, list):
        masked_data["images"] = [_mask_image_payload(image) for image in images]
    elif isinstance(images, tuple):
        masked_data["images"] = tuple(_mask_image_payload(image) for image in images)
    else:
        masked_data["images"] = _mask_image_payload(images)
    return masked_data


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def _generate_teacher_logprobs(
        self,
        client: LLMServerClient,
        teacher_model_config: DistillationTeacherModelConfig,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multi_modal_data = multi_modal_data or {}
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_ids, teacher_logprobs = await self._generate_teacher_logprobs(
            client=client,
            teacher_model_config=teacher_model_config,
            sequence_ids=sequence_ids,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        if (
            self.distillation_loss_config.loss_settings.use_topk
            and self.distillation_loss_config.topk_mode == TOPK_MODE_PROB_PERCEPTION
        ):
            image_data = multi_modal_data.get("images")
            if image_data is None or (isinstance(image_data, (list, tuple)) and len(image_data) == 0):
                raise ValueError("topk_mode='prob_perception' requires at least one image per sample.")
            masked_multi_modal_data = _mask_multi_modal_images(multi_modal_data)
            teacher_ids_mask, teacher_logprobs_mask = await self._generate_teacher_logprobs(
                client=client,
                teacher_model_config=teacher_model_config,
                sequence_ids=sequence_ids,
                multi_modal_data=masked_multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            teacher_ids, teacher_logprobs = build_prob_perception_support_from_topk_logprobs(
                teacher_ids_img=teacher_ids,
                teacher_logprobs_img=teacher_logprobs,
                teacher_ids_mask=teacher_ids_mask,
                teacher_logprobs_mask=teacher_logprobs_mask,
                topk=self.distillation_loss_config.topk,
                perception_candidate_topk=self.distillation_loss_config.perception_candidate_topk,
            )
            teacher_ids = teacher_ids.to(torch.int32)
        return teacher_ids, teacher_logprobs
