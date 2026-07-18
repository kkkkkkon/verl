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
"""Regression guard for verl#6293.

The use_remove_padding=False branch of
FSDPEngineWithLMHead.prepare_model_outputs previously lacked the
distillation_use_topk handling that the use_remove_padding=True branch had,
so distillation outputs were silently dropped from model_output and the
downstream loss raised KeyError. This test invokes prepare_model_outputs on
a stub engine for both branches with distillation_use_topk=True and asserts
the distillation keys produced by logits_processor_func are propagated into
model_output as nested tensors in both cases.

``logprobs_from_logits`` is patched out: in CI environments where flash-attn
is installed, it dispatches to a Triton CrossEntropyLoss kernel that cannot
operate on CPU tensors. The substitute returns a dummy ``log_probs`` tensor
of the right shape, which is sufficient for this test — the contract under
test is the propagation of distillation keys, not the numerical correctness
of log-prob computation.
"""

import asyncio
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_fsdp_forward_kl_topk
from verl.trainer.distillation.losses import compute_forward_kl_topk as collect_forward_kl_topk_metrics
from verl.trainer.distillation.topk_support import (
    build_prob_perception_support_from_logits,
    build_prob_perception_support_from_topk_logprobs,
)
from verl.trainer.ppo.offline_response import (
    apply_offline_chat_template,
    process_offline_multi_modal_info,
    split_offline_response_tokens,
    tokenize_offline_response,
)
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = ("distillation_losses", "student_mass", "overlap_count", "overlap_token_advantage")


def test_split_offline_response_tokens_uses_generation_prompt_boundary():
    prompt_ids = [10, 11, 12]
    full_ids = [10, 11, 12, 20, 21, 22]

    assert split_offline_response_tokens(prompt_ids, full_ids, max_response_length=2) == [20, 21]

    with pytest.raises(ValueError, match="does not preserve the prompt generation prefix"):
        split_offline_response_tokens(prompt_ids, [10, 99, 12, 20], max_response_length=4)


def test_process_offline_multi_modal_info_uses_dataset_protocol_for_multiple_images():
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}]
    data_config = {"image_min_pixels": 100}
    calls = []

    class _Processor:
        image_processor = SimpleNamespace(patch_size=28)

    class _Dataset:
        @classmethod
        async def process_multi_modal_info(cls, received_messages, image_patch_size, config):
            calls.append((received_messages, image_patch_size, config))
            return ["image-1", "image-2"], None, None

    result = asyncio.run(
        process_offline_multi_modal_info(
            messages,
            processor=_Processor(),
            dataset_cls=_Dataset,
            data_config=data_config,
        )
    )

    assert result == {"images": ["image-1", "image-2"]}
    assert calls == [(messages, 28, data_config)]


def test_apply_offline_chat_template_builds_prompt_and_response_without_agent_loop():
    messages = [{"role": "user", "content": "question"}]

    class _Tokenizer:
        def apply_chat_template(
            self, received_messages, *, tokenize, add_generation_prompt, tools, return_dict, **kwargs
        ):
            assert tokenize is True
            assert tools is None
            assert return_dict is False
            if add_generation_prompt:
                return [10, 11, 12]
            assert received_messages[-1]["role"] == "assistant"
            return [10, 11, 12, 20, 21]

    tokenizer = _Tokenizer()
    prompt_ids = asyncio.run(
        apply_offline_chat_template(
            messages,
            tokenizer=tokenizer,
            processor=None,
            add_generation_prompt=True,
        )
    )
    full_ids = asyncio.run(
        apply_offline_chat_template(
            [*messages, {"role": "assistant", "content": "response"}],
            tokenizer=tokenizer,
            processor=None,
            add_generation_prompt=False,
        )
    )

    assert split_offline_response_tokens(prompt_ids, full_ids, max_response_length=4) == [20, 21]


def test_tokenize_offline_response_tokenizes_rendered_assistant_suffix_independently():
    class _Processor:
        def apply_chat_template(
            self, received_messages, *, tokenize, add_generation_prompt, tools, return_dict, **kwargs
        ):
            assert tokenize is False
            assert tools is None
            assert return_dict is False
            if add_generation_prompt:
                return "prompt<assistant>"
            assert received_messages[-1] == {"role": "assistant", "content": "response"}
            return "prompt<assistant>response<end>"

    class _Tokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, text, *, add_special_tokens):
            self.calls.append((text, add_special_tokens))
            return {"input_ids": [20, 21]}

    tokenizer = _Tokenizer()
    response_ids = asyncio.run(
        tokenize_offline_response(
            [{"role": "user", "content": "question"}],
            "response",
            tokenizer=tokenizer,
            processor=_Processor(),
        )
    )

    assert response_ids == [20, 21]
    assert tokenizer.calls == [("response<end>", False)]


def test_distillation_padding_teacher_outputs_match_minimal_sequence():
    support_size = 4
    source_td = {
        "position_ids": torch.arange(6),
        "teacher_ids": torch.randint(0, _VOCAB_SIZE, (6, support_size), dtype=torch.int32),
        "teacher_logprobs": torch.randn(6, support_size),
    }
    template, tag = construct_minimal_padding_template(
        source_td,
        {"prompt_len": 3, "response_len": 3, "seq_len": 6},
        eos_token_id=1,
    )

    assert template["input_ids"].shape == (2,)
    assert template["teacher_ids"].shape == (2, support_size)
    assert template["teacher_logprobs"].shape == (2, support_size)
    assert tag["is_padding"] is True

    teacher_ids = torch.nested.as_nested_tensor([template["teacher_ids"]], layout=torch.jagged)
    teacher_logprobs = torch.nested.as_nested_tensor([template["teacher_logprobs"]], layout=torch.jagged)
    student_logits = torch.randn(1, 2, _VOCAB_SIZE)
    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            use_chunked_topk=False,
            loss_mode="mixed_kl",
            use_tail_bucket=None,
            topk_mode="prob",
            log_prob_min_clamp=-10.0,
            kl_mix_alpha=0.5,
        )
    )

    outputs = compute_fsdp_forward_kl_topk(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )
    assert all(output.shape == (1, 2) for output in outputs.values())
    assert all(torch.isfinite(output).all() for output in outputs.values())


def _make_engine_stub():
    """Bypass FSDPEngineWithLMHead.__init__; set only attributes that
    prepare_model_outputs touches in this test path (no SP, no fused kernels,
    no entropy)."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False

    class _EngineCfg:
        entropy_checkpointing = False

    eng.engine_config = _EngineCfg()
    return eng


def _make_logits_processor(keys):
    """Fake top-k distillation processor: returns one (1, total_nnz) tensor per key.

    The real processor (verl/trainer/distillation/losses.py) returns
    student_logits.shape[:2]; we mimic that contract.
    """

    def _proc(student_logits, data):
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("distillation_only", [False, True])
def test_distillation_outputs_emitted_in_both_padding_modes(use_remove_padding, distillation_only):
    """distillation_use_topk=True must populate distillation outputs into
    model_output regardless of use_remove_padding. See verl#6293.

    When distillation_only=True, log_probs must be omitted (supervised top-k path)."""
    bsz = 2
    seq_lengths_list = [3, 2]
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int64)
    total_nnz = int(seq_lengths.sum())

    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)]).to(torch.int64)

    flat_input_ids = torch.randint(0, _VOCAB_SIZE, (total_nnz,))
    input_ids_nested = torch.nested.nested_tensor_from_jagged(flat_input_ids, offsets=cu_seqlens)

    input_ids_rmpad_rolled = torch.randint(0, _VOCAB_SIZE, (total_nnz,))

    class _Output:
        pass

    output = _Output()

    if use_remove_padding:
        # True branch: output.logits shape (1, total_nnz, V), squeeze(0) -> (total_nnz, V).
        output.logits = torch.randn(1, total_nnz, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature_rmpad": torch.ones(total_nnz),
        }
    else:
        # False branch: output.logits shape (bsz, max_seqlen, V).
        max_seqlen = max(seq_lengths_list)
        output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature": torch.ones(bsz),
        }

    micro_batch = TensorDict({"input_ids": input_ids_nested}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=distillation_only,
        max_response_length=max(seq_lengths_list),
    )

    eng = _make_engine_stub()

    # Patch logprobs_from_logits because flash-attn's Triton CrossEntropyLoss
    # cannot operate on CPU tensors. The shape is what downstream code asserts
    # against (v.shape == log_probs.shape), and prepare_model_outputs reduces
    # both branches to a (total_nnz,) log_probs over the rmpad'ed logits.
    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits",
        return_value=torch.zeros(total_nnz),
    ):
        model_output = FSDPEngineWithLMHead.prepare_model_outputs(
            eng,
            output=output,
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS),
        )

    if distillation_only:
        assert "log_probs" not in model_output, (
            f"log_probs should be omitted when distillation_only=True "
            f"(use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )
    else:
        assert "log_probs" in model_output, (
            f"log_probs missing (use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )

    for k in _DISTILLATION_KEYS:
        assert k in model_output, (
            f"Distillation key '{k}' missing from model_output "
            f"(use_remove_padding={use_remove_padding}); "
            f"keys: {list(model_output.keys())}"
        )
        assert model_output[k].is_nested, (
            f"Expected '{k}' to be a nested tensor (use_remove_padding={use_remove_padding}); "
            f"got {type(model_output[k])}"
        )


def _nested_from_rows(rows):
    values = torch.tensor(rows)
    offsets = torch.tensor([0, len(rows)], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def _nested_from_dense_topk(tensor):
    values = tensor.reshape(-1, tensor.shape[-1])
    offsets = torch.tensor([0, values.shape[0]], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def test_prob_perception_support_reverse_and_mixed_kl_are_finite():
    torch.manual_seed(7)
    bsz, seq_len, vocab_size, topk = 2, 3, 192, 32
    half_topk = topk // 2
    candidate_topk = 128

    student_logits = torch.randn(bsz, seq_len, vocab_size)
    teacher_logits_img = torch.randn(bsz, seq_len, vocab_size)
    teacher_logits_mask = torch.randn(bsz, seq_len, vocab_size)
    # This token has a huge image-minus-mask score but negligible normal-image
    # probability. Candidate restriction must keep it out of the support.
    teacher_logits_img[..., -1] = -50.0
    teacher_logits_mask[..., -1] = -1000.0

    support_ids, teacher_support_logprobs = build_prob_perception_support_from_logits(
        teacher_logits_img=teacher_logits_img,
        teacher_logits_mask=teacher_logits_mask,
        topk=topk,
        perception_candidate_topk=candidate_topk,
    )

    assert support_ids.shape == (bsz, seq_len, topk)
    assert teacher_support_logprobs.shape == (bsz, seq_len, topk)
    candidate_logits, candidate_ids = torch.topk(teacher_logits_img, k=candidate_topk, dim=-1)
    expected_prob_ids = candidate_ids[..., :half_topk]
    candidate_mask_logits = torch.gather(teacher_logits_mask, dim=-1, index=candidate_ids)
    candidate_perception_score = candidate_logits - candidate_mask_logits
    candidate_perception_score[..., :half_topk] = float("-inf")
    expected_perception_positions = torch.topk(candidate_perception_score, k=half_topk, dim=-1).indices
    expected_perception_ids = torch.gather(candidate_ids, dim=-1, index=expected_perception_positions)
    assert torch.equal(support_ids[..., :half_topk], expected_prob_ids)
    assert torch.equal(support_ids[..., half_topk:], expected_perception_ids)
    assert not (support_ids == vocab_size - 1).any()
    sorted_support_ids = torch.sort(support_ids, dim=-1).values
    assert (sorted_support_ids[..., 1:] != sorted_support_ids[..., :-1]).all()

    output = compute_fsdp_forward_kl_topk(
        student_logits=student_logits.reshape(1, bsz * seq_len, vocab_size),
        teacher_topk_log_probs=_nested_from_dense_topk(teacher_support_logprobs),
        teacher_topk_ids=_nested_from_dense_topk(support_ids),
        config=SimpleNamespace(
            distillation_loss=SimpleNamespace(
                log_prob_min_clamp=None,
                loss_mode="reverse_kl",
                topk_mode="prob_perception",
                use_tail_bucket=None,
                use_chunked_topk=False,
            )
        ),
        data_format="thd",
    )

    assert output["distillation_losses"].shape == (1, bsz * seq_len)
    assert torch.isfinite(output["distillation_losses"]).all()
    assert (output["student_mass"] < 1.0).all()
    assert (output["teacher_mass"] < 1.0).all()
    assert torch.isfinite(output["perception_teacher_mass"]).all()

    student_support_logprobs = torch.gather(
        torch.log_softmax(student_logits, dim=-1), dim=-1, index=support_ids
    )
    student_mass = student_support_logprobs.exp().sum(dim=-1)
    teacher_mass = teacher_support_logprobs.exp().sum(dim=-1)
    support_reverse_kl = (
        student_support_logprobs.exp() * (student_support_logprobs - teacher_support_logprobs)
    ).sum(dim=-1)
    eps = torch.finfo(torch.float32).eps
    student_tail = (1.0 - student_mass).clamp_min(eps)
    teacher_tail = (1.0 - teacher_mass).clamp_min(eps)
    expected_reverse_kl = support_reverse_kl + student_tail * (student_tail.log() - teacher_tail.log())
    torch.testing.assert_close(
        output["distillation_losses"].reshape(bsz, seq_len), expected_reverse_kl, rtol=1e-5, atol=1e-6
    )

    mix_alpha = 0.3
    mixed_output = compute_fsdp_forward_kl_topk(
        student_logits=student_logits.reshape(1, bsz * seq_len, vocab_size),
        teacher_topk_log_probs=_nested_from_dense_topk(teacher_support_logprobs),
        teacher_topk_ids=_nested_from_dense_topk(support_ids),
        config=SimpleNamespace(
            distillation_loss=SimpleNamespace(
                log_prob_min_clamp=None,
                loss_mode="mixed_kl",
                kl_mix_alpha=mix_alpha,
                topk_mode="prob_perception",
                use_tail_bucket=None,
                use_chunked_topk=False,
            )
        ),
        data_format="thd",
    )
    support_forward_kl = (
        teacher_support_logprobs.exp() * (teacher_support_logprobs - student_support_logprobs)
    ).sum(dim=-1)
    expected_forward_kl = support_forward_kl + teacher_tail * (teacher_tail.log() - student_tail.log())
    expected_mixed_kl = mix_alpha * expected_forward_kl + (1.0 - mix_alpha) * expected_reverse_kl
    torch.testing.assert_close(
        mixed_output["forward_kl_losses"].reshape(bsz, seq_len), expected_forward_kl, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        mixed_output["reverse_kl_losses"].reshape(bsz, seq_len), expected_reverse_kl, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        mixed_output["distillation_losses"].reshape(bsz, seq_len), expected_mixed_kl, rtol=1e-5, atol=1e-6
    )


def test_prob_perception_runtime_support_is_unique_and_fixed_width():
    torch.manual_seed(11)
    bsz, seq_len, vocab_size = 2, 3, 256
    topk, candidate_topk = 32, 128
    teacher_logits_img = torch.randn(bsz, seq_len, vocab_size)
    teacher_logits_mask = torch.randn(bsz, seq_len, vocab_size)
    teacher_logprobs_img = torch.log_softmax(teacher_logits_img, dim=-1)
    teacher_logprobs_mask = torch.log_softmax(teacher_logits_mask, dim=-1)
    img_logprobs, img_ids = torch.topk(teacher_logprobs_img, k=candidate_topk, dim=-1)
    mask_logprobs, mask_ids = torch.topk(teacher_logprobs_mask, k=candidate_topk, dim=-1)

    support_ids, support_logprobs = build_prob_perception_support_from_topk_logprobs(
        teacher_ids_img=img_ids,
        teacher_logprobs_img=img_logprobs,
        teacher_ids_mask=mask_ids,
        teacher_logprobs_mask=mask_logprobs,
        topk=topk,
        perception_candidate_topk=candidate_topk,
    )

    assert support_ids.shape == (bsz, seq_len, topk)
    assert support_logprobs.shape == (bsz, seq_len, topk)
    sorted_support_ids = torch.sort(support_ids, dim=-1).values
    assert (sorted_support_ids[..., 1:] != sorted_support_ids[..., :-1]).all()
    assert torch.equal(support_ids[..., : topk // 2], img_ids[..., : topk // 2])
    assert torch.allclose(
        support_logprobs,
        torch.gather(teacher_logprobs_img, dim=-1, index=support_ids),
    )


def test_forward_kl_topk_emits_overlap_metrics():
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 1.0, 0.0, 0.0],
            [8.0, 7.0, 0.0, 0.0, 9.0, 0.0],
            [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1, 2], [4, 5], [3, 4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [
            [torch.log(torch.tensor(0.7)), torch.log(torch.tensor(0.2))],
            [torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.3))],
            [torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.4))],
        ]
    ).to(torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None))

    output = compute_fsdp_forward_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )

    torch.testing.assert_close(output["overlap_count"], torch.tensor([[2, 1, 0]]))

    student_log_probs = torch.log_softmax(logits, dim=-1)
    gathered_student = torch.gather(student_log_probs, dim=-1, index=teacher_ids.values().unsqueeze(0))
    teacher_log_probs = teacher_logprobs.values().unsqueeze(0)
    token_adv = -(teacher_log_probs.exp() * (teacher_log_probs - gathered_student))
    expected_ota = torch.tensor(
        [[token_adv[0, 0].mean(), token_adv[0, 1, 0], 0.0]],
        dtype=output["overlap_token_advantage"].dtype,
    )
    torch.testing.assert_close(output["overlap_token_advantage"], expected_ota)


def test_forward_kl_topk_metric_aggregation_for_overlap_outputs():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([0.1, 0.2, 0.3]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([2, 1, 0]),
        "overlap_token_advantage": torch.tensor([-0.2, -0.4, 0.0]),
        "forward_kl_losses": torch.tensor([1.0, 2.0, 9.0]),
        "reverse_kl_losses": torch.tensor([3.0, 4.0, 9.0]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2))

    _, metrics = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )

    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(-0.3)
    assert metrics["distillation/forward_kl_component"] == pytest.approx(1.5)
    assert metrics["distillation/reverse_kl_component"] == pytest.approx(3.5)
