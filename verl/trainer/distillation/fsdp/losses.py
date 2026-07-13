# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    Returns:
        [B, T, K] tensor with the same dtype as `logits`.
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)

    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _gather_log_probs_without_full_softmax(logits: torch.Tensor, support_ids: torch.Tensor) -> torch.Tensor:
    """Gather exact support log-probs without materializing another [B, T, V] tensor."""
    support_logits = torch.gather(logits, dim=-1, index=support_ids)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    return support_logits - log_z


def _tail_bucket_kl(
    student_mass: torch.Tensor,
    teacher_mass: torch.Tensor,
    reverse: bool,
) -> torch.Tensor:
    """KL contribution of all support-excluded tokens aggregated as one bucket."""
    eps = torch.finfo(torch.float32).eps
    student_tail = (1.0 - student_mass.float()).clamp(min=eps, max=1.0)
    teacher_tail = (1.0 - teacher_mass.float()).clamp(min=eps, max=1.0)
    if reverse:
        return student_tail * (student_tail.log() - teacher_tail.log())
    return teacher_tail * (teacher_tail.log() - student_tail.log())


def _effective_use_tail_bucket(loss_config: DistillationLossConfig, loss_mode: str) -> bool:
    configured = getattr(loss_config, "use_tail_bucket", None)
    if configured is not None:
        return configured
    return getattr(loss_config, "topk_mode", "prob") == "prob_perception" or loss_mode in {
        "reverse_kl",
        "mixed_kl",
        "forward_reverse_kl",
    }


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute forward, reverse, or mixed KL using top-k support log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    loss_mode = getattr(loss_config, "loss_mode", "forward_kl_topk")
    mixed_kl = loss_mode in {"mixed_kl", "forward_reverse_kl"}
    reverse_kl = loss_mode == "reverse_kl"
    use_tail_bucket = _effective_use_tail_bucket(loss_config, loss_mode)
    raw_teacher_support_log_probs = teacher_topk_log_probs.float()
    teacher_mass = raw_teacher_support_log_probs.exp().sum(dim=-1)
    student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices

    if use_chunked_topk:
        raw_student_support_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        ).float()
    elif use_tail_bucket or reverse_kl or mixed_kl:
        raw_student_support_log_probs = _gather_log_probs_without_full_softmax(
            student_logits, teacher_topk_ids
        ).float()
    else:
        # Preserve the original eager forward-KL path when no tail bucket is
        # requested. Perception/reverse modes avoid this full-size buffer.
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        raw_student_support_log_probs = torch.gather(
            student_log_probs, dim=-1, index=teacher_topk_ids
        ).float()

    student_mass = raw_student_support_log_probs.exp().sum(dim=-1)
    student_support_log_probs = raw_student_support_log_probs
    teacher_support_log_probs = raw_teacher_support_log_probs
    if loss_config.log_prob_min_clamp is not None:
        student_support_log_probs = student_support_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_support_log_probs = teacher_support_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    forward_kl_losses = None
    reverse_kl_losses = None
    forward_token_kl = None
    reverse_token_kl = None
    if use_tail_bucket:
        teacher_support_probs = raw_teacher_support_log_probs.exp()
        student_support_probs = raw_student_support_log_probs.exp()
        forward_token_kl = teacher_support_probs * (
            teacher_support_log_probs - student_support_log_probs
        )
        reverse_token_kl = student_support_probs * (
            student_support_log_probs - teacher_support_log_probs
        )
        forward_kl_losses = forward_token_kl.sum(dim=-1) + _tail_bucket_kl(
            student_mass=student_mass,
            teacher_mass=teacher_mass,
            reverse=False,
        )
        reverse_kl_losses = reverse_token_kl.sum(dim=-1) + _tail_bucket_kl(
            student_mass=student_mass,
            teacher_mass=teacher_mass,
            reverse=True,
        )
    elif reverse_kl or mixed_kl:
        # Without a tail bucket, compare distributions conditioned on support S.
        student_cond_log_probs = F.log_softmax(student_support_log_probs, dim=-1)
        teacher_cond_log_probs = F.log_softmax(teacher_support_log_probs, dim=-1)
        student_cond_probs = student_cond_log_probs.exp()
        teacher_cond_probs = teacher_cond_log_probs.exp()
        forward_token_kl = teacher_cond_probs * (
            teacher_cond_log_probs - student_cond_log_probs
        )
        reverse_token_kl = student_cond_probs * (
            student_cond_log_probs - teacher_cond_log_probs
        )
        forward_kl_losses = forward_token_kl.sum(dim=-1)
        reverse_kl_losses = reverse_token_kl.sum(dim=-1)
    else:
        # Preserve the original truncated forward-KL behavior and clamped
        # probability weighting when no tail bucket is requested.
        teacher_support_probs = teacher_support_log_probs.exp()
        forward_token_kl = teacher_support_probs * (
            teacher_support_log_probs - student_support_log_probs
        )
        forward_kl_losses = forward_token_kl.sum(dim=-1)

    if mixed_kl:
        mix_alpha = float(getattr(loss_config, "kl_mix_alpha", 0.5))
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError(f"kl_mix_alpha must be in [0, 1], got {mix_alpha}.")
        assert forward_kl_losses is not None and reverse_kl_losses is not None
        assert forward_token_kl is not None and reverse_token_kl is not None
        distillation_losses = mix_alpha * forward_kl_losses + (1.0 - mix_alpha) * reverse_kl_losses
        token_kl = mix_alpha * forward_token_kl + (1.0 - mix_alpha) * reverse_token_kl
    elif reverse_kl:
        assert reverse_kl_losses is not None and reverse_token_kl is not None
        distillation_losses = reverse_kl_losses
        token_kl = reverse_token_kl
    else:
        assert forward_kl_losses is not None and forward_token_kl is not None
        distillation_losses = forward_kl_losses
        token_kl = forward_token_kl

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
    if mixed_kl:
        outputs["forward_kl_losses"] = forward_kl_losses
        outputs["reverse_kl_losses"] = reverse_kl_losses
    if getattr(loss_config, "topk_mode", "prob") == "prob_perception":
        perception_start = teacher_topk_ids.shape[-1] // 2
        outputs["perception_teacher_mass"] = raw_teacher_support_log_probs[..., perception_start:].exp().sum(dim=-1)
    return outputs
