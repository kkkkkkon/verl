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
"""Runtime adapter for raw image QA parquet rows.

The base RLHFDataset already handles parquet loading, concatenation, length
filtering, and collation. This adapter maps raw image QA rows into the
chat-message fields expected by verl and optionally carries an offline model
response for direct distillation.
"""

import os
import re
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset


DEFAULT_DATA_SOURCE = "PAPOGalaxy/PAPO_ViRL39K_train"
DEFAULT_INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE enclosed within <answer> </answer> tags."
)
DEFAULT_PROMPT_TEMPLATE = "<image>\n{problem}\n\n{instruction}"
PROMPT_INSTRUCTION_ENV = "RAW_IMAGE_QA_PROMPT_INSTRUCTION"
PROMPT_TEMPLATE_ENV = "RAW_IMAGE_QA_PROMPT_TEMPLATE"


class RawImageQADataset(RLHFDataset):
    """Adapt raw image/question/answer rows to verl chat-message rows."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            config = args[2]
        data_files = kwargs.get("data_files")
        if data_files is None and args:
            data_files = args[0]

        self.answer_key = config.get("answer_key", "answer") if config is not None else "answer"
        self.response_key = config.get("response_key", "response") if config is not None else "response"
        self.default_data_source = self._resolve_default_data_source(config, data_files)
        self.image_min_pixels = (
            int(config.get("image_min_pixels"))
            if config is not None and config.get("image_min_pixels") is not None
            else None
        )
        self.image_max_pixels = (
            int(config.get("image_max_pixels"))
            if config is not None and config.get("image_max_pixels") is not None
            else None
        )
        if (
            self.image_min_pixels is not None
            and self.image_max_pixels is not None
            and self.image_min_pixels > self.image_max_pixels
        ):
            raise ValueError("image_min_pixels cannot be larger than image_max_pixels.")

        env_instruction = os.environ.get(PROMPT_INSTRUCTION_ENV)
        env_template = os.environ.get(PROMPT_TEMPLATE_ENV)
        self.prompt_instruction = (
            config.get("prompt_instruction", env_instruction or DEFAULT_INSTRUCTION)
            if config is not None
            else env_instruction or DEFAULT_INSTRUCTION
        )
        self.prompt_template = (
            config.get("prompt_template", env_template or DEFAULT_PROMPT_TEMPLATE)
            if config is not None
            else env_template or DEFAULT_PROMPT_TEMPLATE
        )

        super().__init__(*args, **kwargs)

    @staticmethod
    def _normalize_file_list(files: Any) -> list[str]:
        if files is None:
            return []
        if isinstance(files, str | os.PathLike):
            return [os.fspath(files)]
        return [os.fspath(file) for file in files]

    @classmethod
    def _resolve_default_data_source(cls, config, data_files: Any) -> str:
        if config is None:
            return DEFAULT_DATA_SOURCE

        default_data_source = config.get("default_data_source", DEFAULT_DATA_SOURCE)
        val_default_data_source = config.get("val_default_data_source", default_data_source)
        if sorted(cls._normalize_file_list(data_files)) == sorted(cls._normalize_file_list(config.get("val_files"))):
            return val_default_data_source
        return default_data_source

    @staticmethod
    def _as_image_list(image_payload: Any) -> list[Any]:
        if image_payload is None:
            return []
        if isinstance(image_payload, list):
            return [image for image in image_payload if image is not None]
        return [image_payload]

    @staticmethod
    def _to_image_content(image_payload: Any) -> dict[str, Any]:
        if isinstance(image_payload, Image.Image):
            return {"type": "image", "image": image_payload.convert("RGB")}
        if isinstance(image_payload, bytes):
            return {"type": "image", "image": Image.open(BytesIO(image_payload)).convert("RGB")}
        if isinstance(image_payload, dict):
            payload = dict(image_payload)
            if payload.get("bytes") is not None:
                payload["image"] = Image.open(BytesIO(payload["bytes"])).convert("RGB")
            elif payload.get("path") is not None and "image" not in payload:
                payload["image"] = payload["path"]
            payload["type"] = "image"
            return payload
        if isinstance(image_payload, str | os.PathLike):
            return {"type": "image", "image": os.fspath(image_payload)}
        raise TypeError(f"Unsupported image type: {type(image_payload)}")

    def _to_limited_image_content(self, image_payload: Any) -> dict[str, Any]:
        content = self._to_image_content(image_payload)
        if self.image_min_pixels is not None:
            content.setdefault("min_pixels", self.image_min_pixels)
        if self.image_max_pixels is not None:
            content.setdefault("max_pixels", self.image_max_pixels)
        return content

    def _render_prompt(self, problem: Any) -> str:
        problem = str(problem).replace("<image>", "").strip()
        return self.prompt_template.format(problem=problem, instruction=self.prompt_instruction).strip()

    def _get_image_payload(self, example: dict) -> Any:
        if self.image_key in example and example[self.image_key] is not None:
            return example[self.image_key]
        # Keep both the original PAPO `image` column and datasets that expose
        # an `images` list usable without rewriting parquet files.
        fallback_key = "image" if self.image_key == "images" else "images"
        return example.get(fallback_key)

    def _build_messages(self, example: dict, key: str):
        prompt_text = self._render_prompt(example.get(key, ""))
        images = self._as_image_list(self._get_image_payload(example))
        image_slot_count = prompt_text.count("<image>")
        image_offset = 0
        content = []

        for segment in re.split("(<image>)", prompt_text):
            if not segment:
                continue
            if segment == "<image>":
                # The default template has one visual slot. When a row contains
                # multiple images, place all of them at that slot in source order.
                if image_slot_count == 1:
                    images_for_slot = images[image_offset:]
                else:
                    images_for_slot = images[image_offset : image_offset + 1]
                content.extend(self._to_limited_image_content(image) for image in images_for_slot)
                image_offset += len(images_for_slot)
                continue
            content.append({"type": "text", "text": segment})

        if image_offset == 0 and images:
            content = [self._to_limited_image_content(image) for image in images] + content
            image_offset = len(images)
        for image in images[image_offset:]:
            content.append(self._to_limited_image_content(image))

        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_media(messages: list[dict], media_type: str) -> list[Any] | None:
        values = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != media_type:
                    continue
                if media_type in item:
                    values.append(item[media_type])
                elif media_type == "image" and "image_url" in item:
                    values.append(item["image_url"])
                else:
                    values.append({k: v for k, v in item.items() if k != "type"})
        return values or None

    @classmethod
    def _process_multi_modal_info(cls, messages: list[dict], image_patch_size, config):
        if hasattr(super(), "_process_multi_modal_info"):
            return super()._process_multi_modal_info(messages, image_patch_size, config)

        images = cls._extract_media(messages, "image")
        videos = cls._extract_media(messages, "video")
        audios = cls._extract_media(messages, "audio")
        return images, videos, audios

    @classmethod
    async def process_multi_modal_info(cls, messages: list[dict], image_patch_size, config):
        return cls._process_multi_modal_info(messages, image_patch_size, config)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        answer = row_dict.get(self.answer_key, "")
        response = row_dict.get(self.response_key)
        if response is None and self.config.get("response_key") is not None:
            raise KeyError(f"Dataset row {item} is missing response field {self.response_key!r}.")
        extra_info = row_dict.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info.setdefault("index", item)
        extra_info.setdefault("question", str(row_dict.get(self.prompt_key, "")))
        extra_info.setdefault("answer", str(answer))

        output = {
            "raw_prompt": self._build_messages(row_dict, key=self.prompt_key),
            "data_source": row_dict.get("data_source", self.default_data_source),
            "ability": row_dict.get("ability", "visual_reasoning"),
            "reward_model": row_dict.get("reward_model", {"style": "rule", "ground_truth": str(answer)}),
            "extra_info": extra_info,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": extra_info.get("index", item),
            "tools_kwargs": extra_info.get("tools_kwargs", {}),
            "interaction_kwargs": extra_info.get("interaction_kwargs", {}),
        }
        if response is not None:
            output[self.response_key] = str(response)
        return output
