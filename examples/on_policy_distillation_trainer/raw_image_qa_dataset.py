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

The raw parquet is expected to contain the only fields we need:
  - image: original image payload, not a path
  - problem: question text
  - answer: target answer

This class reuses verl's RLHFDataset loading/filtering/collation path and only
maps one raw row into the runtime fields expected by the agent loop.
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
    """Adapt raw image QA rows to verl rollout format at read time."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            config = args[2]

        self.answer_key = config.get("answer_key", "answer") if config is not None else "answer"
        self.default_data_source = (
            config.get("default_data_source", DEFAULT_DATA_SOURCE) if config is not None else DEFAULT_DATA_SOURCE
        )
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

    def _render_prompt(self, problem: Any) -> str:
        problem = str(problem).replace("<image>", "").strip()
        return self.prompt_template.format(problem=problem, instruction=self.prompt_instruction).strip()

    def _build_messages(self, example: dict, key: str):
        prompt_text = self._render_prompt(example.get(key, ""))
        images = self._as_image_list(example.get(self.image_key, None))
        image_offset = 0
        content = []

        for segment in re.split("(<image>)", prompt_text):
            if not segment:
                continue
            if segment == "<image>":
                if image_offset < len(images):
                    content.append(self._to_image_content(images[image_offset]))
                    image_offset += 1
                continue
            content.append({"type": "text", "text": segment})

        if image_offset == 0 and images:
            content = [self._to_image_content(images[0]), *content]
            image_offset = 1
        for image in images[image_offset:]:
            content.insert(0, self._to_image_content(image))

        return [{"role": "user", "content": content}]

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        answer = row_dict.get(self.answer_key, "")
        extra_info = row_dict.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info.setdefault("index", item)
        extra_info.setdefault("question", str(row_dict.get(self.prompt_key, "")))
        extra_info.setdefault("answer", str(answer))

        return {
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
