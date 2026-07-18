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

import asyncio
from typing import Any

from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template as render_chat_template


async def apply_offline_chat_template(
    messages: list[dict],
    *,
    tokenizer: Any,
    processor: Any,
    add_generation_prompt: bool,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    images: Any = None,
    videos: Any = None,
    audios: Any = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Tokenize an offline prompt or prompt/response chat without an agent loop."""
    chat_template_kwargs = apply_chat_template_kwargs or {}
    if processor is not None:
        raw_sequence = await asyncio.to_thread(
            render_chat_template,
            processor,
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **chat_template_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            processor,
            text=[raw_sequence],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return normalize_token_ids(model_inputs.pop("input_ids"))

    tokenized_sequence = await asyncio.to_thread(
        render_chat_template,
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **chat_template_kwargs,
    )
    return normalize_token_ids(tokenized_sequence)


async def tokenize_offline_response(
    messages: list[dict],
    response: Any,
    *,
    tokenizer: Any,
    processor: Any,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Render and tokenize only the assistant continuation of an offline chat.

    Tokenizing the prompt and the full prompt/response chat independently does
    not guarantee that the prompt token IDs are a prefix of the full token IDs:
    a tokenizer may merge text across the generation boundary.  Autoregressive
    generation cannot change the already-tokenized prompt, so tokenize the
    rendered assistant suffix independently instead.
    """
    chat_template_kwargs = apply_chat_template_kwargs or {}
    template_source = processor if processor is not None else tokenizer
    full_messages = [*messages, {"role": "assistant", "content": str(response)}]

    prompt_text = render_chat_template(
        template_source,
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **chat_template_kwargs,
    )
    full_text = render_chat_template(
        template_source,
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
        **chat_template_kwargs,
    )
    if not isinstance(prompt_text, str) or not isinstance(full_text, str):
        raise TypeError(
            "Offline chat templates must render strings when tokenize=False, "
            f"got {type(prompt_text).__name__} and {type(full_text).__name__}."
        )
    if not full_text.startswith(prompt_text):
        mismatch = next(
            (
                i
                for i, (prompt_char, full_char) in enumerate(zip(prompt_text, full_text, strict=False))
                if prompt_char != full_char
            ),
            min(len(prompt_text), len(full_text)),
        )
        raise ValueError(
            "The full chat template does not preserve the prompt generation prefix; "
            f"first character mismatch at {mismatch}."
        )

    response_text = full_text[len(prompt_text) :]
    tokenized_response = tokenizer(response_text, add_special_tokens=False)
    response_ids = normalize_token_ids(tokenized_response)
    if not response_ids:
        raise ValueError("Dataset response produced no assistant tokens after applying the chat template.")
    return response_ids


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
