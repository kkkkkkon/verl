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
"""Validation accuracy metric with no task reward for distillation-only OPD baselines."""

import re
from typing import Any

from mathruler.grader import grade_answer


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def _cleanup_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s*(<|>|/)\s*", r"\1", text)
    return text.strip()


def _extract_answer(response: Any) -> str:
    response = _cleanup_text(response)

    answer_matches = _ANSWER_RE.findall(response)
    if answer_matches:
        return _cleanup_text(answer_matches[-1])

    boxed_matches = _BOXED_RE.findall(response)
    if boxed_matches:
        return _cleanup_text(boxed_matches[-1])

    # Fall back to the final non-empty line so validation still produces a
    # useful metric if the model misses the requested answer tags.
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return _cleanup_text(lines[-1] if lines else response)


def _accuracy(response: Any, ground_truth: Any) -> float:
    prediction = _extract_answer(response)
    target = _cleanup_text(ground_truth)
    if not prediction or not target:
        return 0.0

    try:
        return 1.0 if grade_answer(prediction, target) else 0.0
    except Exception:
        return 1.0 if prediction.lower() == target.lower() else 0.0


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    # These baselines set distillation.distillation_loss.use_task_rewards=False.
    # Keep score at zero so task reward is not mixed into the vanilla
    # distillation objective, while still reporting validation accuracy.
    accuracy_score = _accuracy(solution_str, ground_truth)
    return {
        "score": 0.0,
        "acc": accuracy_score,
        "accuracy": accuracy_score,
        "pred": _extract_answer(solution_str),
    }


def compute_score_wo_format(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        accuracy_score = _accuracy(reward_input.get("response"), reward_input.get("ground_truth"))
        scores.append(
            {
                "overall": accuracy_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
