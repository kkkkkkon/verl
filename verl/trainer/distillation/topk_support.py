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

import torch
import torch.nn.functional as F

TOPK_MODE_PROB = "prob"
TOPK_MODE_PROB_PERCEPTION = "prob_perception"
VALID_TOPK_MODES = {TOPK_MODE_PROB, TOPK_MODE_PROB_PERCEPTION}


def validate_topk_mode(topk_mode: str) -> None:
    if topk_mode not in VALID_TOPK_MODES:
        raise ValueError(
            f"Unsupported distillation topk_mode {topk_mode!r}; expected one of {sorted(VALID_TOPK_MODES)}."
        )


def split_prob_perception_topk(topk: int) -> tuple[int, int]:
    if topk <= 0:
        raise ValueError(f"topk must be positive for prob_perception, got {topk}.")
    if topk % 2 != 0:
        raise ValueError(f"prob_perception requires an even topk so support can be split evenly, got {topk}.")
    prob_topk = topk // 2
    perception_topk = topk - prob_topk
    return prob_topk, perception_topk


def build_topk_support_ids(
    teacher_logits_img: torch.Tensor,
    *,
    topk: int,
    topk_mode: str = TOPK_MODE_PROB,
    teacher_logits_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build distillation support token ids from teacher logits.

    ``prob`` preserves the existing behavior: use teacher full-image probability top-k.
    ``prob_perception`` concatenates half probability top-k ids and half ids from
    ``teacher_logits_img - teacher_logits_mask``. Duplicates are intentionally kept.
    """

    validate_topk_mode(topk_mode)
    if topk_mode == TOPK_MODE_PROB:
        return torch.topk(teacher_logits_img, k=topk, dim=-1).indices

    if teacher_logits_mask is None:
        raise ValueError("teacher_logits_mask is required when topk_mode='prob_perception'.")
    if teacher_logits_mask.shape != teacher_logits_img.shape:
        raise ValueError(
            f"teacher_logits_mask shape must match teacher_logits_img shape, got "
            f"{teacher_logits_mask.shape=} and {teacher_logits_img.shape=}."
        )

    prob_topk, perception_topk = split_prob_perception_topk(topk)
    prob_ids = torch.topk(teacher_logits_img, k=prob_topk, dim=-1).indices
    perception_score = teacher_logits_img - teacher_logits_mask
    perception_ids = torch.topk(perception_score, k=perception_topk, dim=-1).indices
    return torch.cat([prob_ids, perception_ids], dim=-1)


def gather_teacher_logprobs_from_support(teacher_logits_img: torch.Tensor, support_ids: torch.Tensor) -> torch.Tensor:
    """Gather the teacher full-image target distribution on ``support_ids``."""

    teacher_log_probs = F.log_softmax(teacher_logits_img, dim=-1)
    return torch.gather(teacher_log_probs, dim=-1, index=support_ids)


def build_prob_perception_support_from_topk_outputs(
    teacher_ids_img: torch.Tensor,
    teacher_logprobs_img: torch.Tensor,
    teacher_ids_mask: torch.Tensor,
    teacher_logprobs_mask: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build prob_perception support from prompt-logprobs backends.

    Current async teacher servers expose top-k prompt logprobs instead of full
    vocab logits. This helper keeps targets from the full-image forward pass and
    ranks perception ids within the available full-image top-k support.
    """

    if teacher_ids_img.shape != teacher_logprobs_img.shape:
        raise ValueError(
            f"teacher_ids_img and teacher_logprobs_img must have the same shape, got "
            f"{teacher_ids_img.shape} and {teacher_logprobs_img.shape}."
        )
    if teacher_ids_mask.shape != teacher_logprobs_mask.shape:
        raise ValueError(
            f"teacher_ids_mask and teacher_logprobs_mask must have the same shape, got "
            f"{teacher_ids_mask.shape} and {teacher_logprobs_mask.shape}."
        )
    if teacher_ids_img.shape[:-1] != teacher_ids_mask.shape[:-1]:
        raise ValueError(
            f"Full-image and masked teacher outputs must have matching token dimensions, got "
            f"{teacher_ids_img.shape[:-1]} and {teacher_ids_mask.shape[:-1]}."
        )

    prob_topk, perception_topk = split_prob_perception_topk(topk)
    if teacher_ids_img.shape[-1] < topk:
        raise ValueError(f"prob_perception prompt-logprobs path requires at least {topk} full-image ids.")
    if teacher_ids_mask.shape[-1] < 1:
        raise ValueError("prob_perception prompt-logprobs path requires masked-image logprobs.")

    prob_ids = teacher_ids_img[..., :prob_topk]
    prob_logprobs = teacher_logprobs_img[..., :prob_topk]

    matches = teacher_ids_img.unsqueeze(-1) == teacher_ids_mask.unsqueeze(-2)
    missing = torch.full_like(teacher_logprobs_img, -float("inf"))
    masked_logprobs_on_img_support = torch.where(
        matches,
        teacher_logprobs_mask.unsqueeze(-2),
        missing.unsqueeze(-1),
    ).max(dim=-1).values
    perception_score = teacher_logprobs_img - masked_logprobs_on_img_support
    perception_positions = torch.topk(perception_score, k=perception_topk, dim=-1).indices
    perception_ids = torch.gather(teacher_ids_img, dim=-1, index=perception_positions)
    perception_logprobs = torch.gather(teacher_logprobs_img, dim=-1, index=perception_positions)

    return torch.cat([prob_ids, perception_ids], dim=-1), torch.cat([prob_logprobs, perception_logprobs], dim=-1)
