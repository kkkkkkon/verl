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

from typing import Any


async def process_offline_multi_modal_info(
    messages: list[dict], *, processor: Any, dataset_cls: type, data_config: Any
) -> dict[str, Any]:
    """Extract multimodal payloads using the dataset protocol used by agent loops."""
    if processor is None:
        return {}

    image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 14)
    if hasattr(dataset_cls, "process_multi_modal_info"):
        images, videos, audios = await dataset_cls.process_multi_modal_info(
            messages, image_patch_size=image_patch_size, config=data_config
        )
    else:
        images, videos = await dataset_cls.process_vision_info(
            messages, image_patch_size=image_patch_size, config=data_config
        )
        audios = None

    multi_modal_data = {}
    if images is not None:
        multi_modal_data["images"] = images
    if videos is not None:
        multi_modal_data["videos"] = videos
    if audios is not None:
        multi_modal_data["audios"] = audios
    return multi_modal_data


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
