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
    if topk < 2 or topk % 2 != 0:
        raise ValueError(f"prob_perception topk must be an even integer >= 2, got {topk}.")
    prob_topk = topk // 2
    perception_topk = topk - prob_topk
    return prob_topk, perception_topk


def _validate_candidate_topk(candidate_topk: int, topk: int, available: int) -> int:
    if candidate_topk <= topk:
        raise ValueError(
            "perception_candidate_topk must be > topk so perception ranking has a nontrivial candidate pool, "
            f"got {candidate_topk=} and {topk=}."
        )
    if candidate_topk > available:
        raise ValueError(
            f"perception_candidate_topk ({candidate_topk}) exceeds the number of available candidates ({available})."
        )
    return candidate_topk


def _lookup_topk_logprobs_by_id(
    query_ids: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Match top-k token ids without allocating an [..., query_k, reference_k] tensor."""
    sorted_ids, order = torch.sort(reference_ids, dim=-1)
    sorted_logprobs = torch.gather(reference_logprobs, dim=-1, index=order)
    positions = torch.searchsorted(sorted_ids, query_ids)
    safe_positions = positions.clamp_max(sorted_ids.size(-1) - 1)
    found_ids = torch.gather(sorted_ids, dim=-1, index=safe_positions)
    found_logprobs = torch.gather(sorted_logprobs, dim=-1, index=safe_positions)
    matched = (positions < sorted_ids.size(-1)) & (found_ids == query_ids)

    # Tokens missing from the masked top-k lie below its final entry. Using
    # that entry is conservative and avoids promoting an unobserved tail.
    masked_floor = reference_logprobs.min(dim=-1, keepdim=True).values
    return torch.where(matched, found_logprobs, masked_floor.expand_as(found_logprobs))


def build_prob_perception_support_from_logits(
    teacher_logits_img: torch.Tensor,
    teacher_logits_mask: torch.Tensor,
    topk: int,
    perception_candidate_topk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a unique fixed-width support and full-image teacher targets."""
    if teacher_logits_img.shape != teacher_logits_mask.shape:
        raise ValueError(
            "teacher_logits_img and teacher_logits_mask must have the same shape, "
            f"got {teacher_logits_img.shape=} and {teacher_logits_mask.shape=}."
        )
    prob_topk, perception_topk = split_prob_perception_topk(topk)
    candidate_topk = min(perception_candidate_topk, teacher_logits_img.size(-1))
    _validate_candidate_topk(candidate_topk, topk, teacher_logits_img.size(-1))

    candidate_logits_img, candidate_ids = torch.topk(
        teacher_logits_img, k=candidate_topk, dim=-1, sorted=True
    )
    prob_ids = candidate_ids[..., :prob_topk]
    candidate_logits_mask = torch.gather(teacher_logits_mask, dim=-1, index=candidate_ids)
    perception_score = candidate_logits_img - candidate_logits_mask
    perception_score = perception_score.clone()
    perception_score[..., :prob_topk] = float("-inf")
    perception_positions = torch.topk(perception_score, k=perception_topk, dim=-1).indices
    perception_ids = torch.gather(candidate_ids, dim=-1, index=perception_positions)
    support_ids = torch.cat([prob_ids, perception_ids], dim=-1)

    teacher_log_probs_img = F.log_softmax(teacher_logits_img, dim=-1)
    support_log_probs = torch.gather(teacher_log_probs_img, dim=-1, index=support_ids)
    return support_ids, support_log_probs


def build_prob_perception_support_from_topk_logprobs(
    teacher_ids_img: torch.Tensor,
    teacher_logprobs_img: torch.Tensor,
    teacher_ids_mask: torch.Tensor,
    teacher_logprobs_mask: torch.Tensor,
    topk: int,
    perception_candidate_topk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build prob/perception support from backend-returned top-k logprobs.

    Perception candidates are restricted to the normal-image probability top-M.
    Masked logprobs are matched by token id without a quadratic match tensor.
    """
    if teacher_ids_img.shape != teacher_logprobs_img.shape:
        raise ValueError(
            "teacher_ids_img and teacher_logprobs_img must have the same shape, "
            f"got {teacher_ids_img.shape=} and {teacher_logprobs_img.shape=}."
        )
    if teacher_ids_mask.shape != teacher_logprobs_mask.shape:
        raise ValueError(
            "teacher_ids_mask and teacher_logprobs_mask must have the same shape, "
            f"got {teacher_ids_mask.shape=} and {teacher_logprobs_mask.shape=}."
        )
    if teacher_ids_img.shape[:-1] != teacher_ids_mask.shape[:-1]:
        raise ValueError(
            "full-image and masked-image top-k tensors must share leading dims, "
            f"got {teacher_ids_img.shape=} and {teacher_ids_mask.shape=}."
        )

    prob_topk, perception_topk = split_prob_perception_topk(topk)
    candidate_topk = _validate_candidate_topk(perception_candidate_topk, topk, teacher_ids_img.size(-1))
    candidate_logprobs, candidate_positions = torch.topk(
        teacher_logprobs_img, k=candidate_topk, dim=-1, sorted=True
    )
    candidate_ids = torch.gather(teacher_ids_img, dim=-1, index=candidate_positions)
    prob_ids = candidate_ids[..., :prob_topk]
    prob_logprobs = candidate_logprobs[..., :prob_topk]

    matched_mask_logprobs = _lookup_topk_logprobs_by_id(
        query_ids=candidate_ids,
        reference_ids=teacher_ids_mask,
        reference_logprobs=teacher_logprobs_mask,
    )
    perception_score = candidate_logprobs - matched_mask_logprobs
    perception_score = perception_score.clone()
    perception_score[..., :prob_topk] = float("-inf")
    perception_positions = torch.topk(perception_score, k=perception_topk, dim=-1).indices
    perception_ids = torch.gather(candidate_ids, dim=-1, index=perception_positions)
    perception_logprobs = torch.gather(candidate_logprobs, dim=-1, index=perception_positions)

    support_ids = torch.cat([prob_ids, perception_ids], dim=-1)
    support_logprobs = torch.cat([prob_logprobs, perception_logprobs], dim=-1)
    return support_ids, support_logprobs
