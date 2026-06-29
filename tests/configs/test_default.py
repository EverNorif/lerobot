# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import pytest

from lerobot.configs.default import DatasetConfig


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


def test_dataset_config_multi_repo_sample_weights_ok():
    DatasetConfig(repo_id=["user/repo_a", "user/repo_b"], sample_weights=[0.7, 0.3])


def test_dataset_config_sample_weights_require_multi_repo():
    with pytest.raises(ValueError, match="can only be used"):
        DatasetConfig(repo_id="user/repo", sample_weights=[1.0])


def test_dataset_config_sample_weights_match_repo_count():
    with pytest.raises(ValueError, match="same length"):
        DatasetConfig(repo_id=["user/repo_a", "user/repo_b"], sample_weights=[1.0])


def test_dataset_config_sample_weights_positive():
    with pytest.raises(ValueError, match="positive"):
        DatasetConfig(repo_id=["user/repo_a", "user/repo_b"], sample_weights=[0.7, 0.0])


def test_dataset_config_multi_repo_non_empty():
    with pytest.raises(ValueError, match="At least one"):
        DatasetConfig(repo_id=[])
