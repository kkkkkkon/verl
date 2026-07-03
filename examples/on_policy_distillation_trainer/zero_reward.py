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
"""No-op task reward for distillation-only OPD baselines."""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    # These baselines set distillation.distillation_loss.use_task_rewards=False.
    # Returning zero keeps verl's reward pipeline well-defined without mixing
    # task reward into the vanilla distillation objective.
    return {"score": 0.0, "acc": 0.0}
