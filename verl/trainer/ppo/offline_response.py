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


def split_offline_response_tokens(
    prompt_ids: list[int], full_ids: list[int], max_response_length: int
) -> list[int]:
    """Split a templated user/assistant sequence at the generation prompt boundary."""
    if max_response_length <= 0:
        raise ValueError(f"max_response_length must be positive, got {max_response_length}.")
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        mismatch = next(
            (
                i
                for i, (prompt_id, full_id) in enumerate(zip(prompt_ids, full_ids, strict=False))
                if prompt_id != full_id
            ),
            min(len(prompt_ids), len(full_ids)),
        )
        raise ValueError(
            "The full chat template does not preserve the prompt generation prefix; "
            f"first mismatch at token {mismatch}."
        )
    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids:
        raise ValueError("Dataset response produced no assistant tokens after applying the chat template.")
    return response_ids[:max_response_length]
