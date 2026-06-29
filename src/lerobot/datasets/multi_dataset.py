#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torch.utils

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoFrame
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)


def _resize_with_pad_image(image: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    target_channels, target_height, target_width = target_shape
    if image.ndim < 3:
        raise ValueError(f"Expected an image tensor with at least 3 dimensions, got shape {tuple(image.shape)}.")
    if image.shape[-3] != target_channels:
        raise ValueError(
            f"Cannot resize image with {image.shape[-3]} channels to target with {target_channels} channels."
        )
    if tuple(image.shape[-3:]) == target_shape:
        return image

    original_shape = image.shape
    image = image.reshape(-1, *original_shape[-3:])
    _, _, current_height, current_width = image.shape

    ratio = max(current_width / target_width, current_height / target_height)
    resized_height = max(1, int(current_height / ratio))
    resized_width = max(1, int(current_width / ratio))
    resized = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    pad_top, remainder_height = divmod(target_height - resized_height, 2)
    pad_bottom = pad_top + remainder_height
    pad_left, remainder_width = divmod(target_width - resized_width, 2)
    pad_right = pad_left + remainder_width
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

    return padded.reshape(*original_shape[:-2], target_height, target_width)


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        if len(self._datasets) == 0:
            raise ValueError("At least one dataset must be provided.")

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logger.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
        self.meta = deepcopy(self._datasets[0].meta)
        self.meta.stats = self.stats
        self.meta.info["total_frames"] = self.num_frames
        self.meta.info["total_episodes"] = self.num_episodes
        self._camera_resize_shapes = self._get_camera_resize_shapes()
        episode_offsets = []
        frame_offset = 0
        for dataset in self._datasets:
            episodes = dataset.meta.episodes
            offset_episodes = episodes.map(
                lambda batch, offset=frame_offset: {
                    "dataset_from_index": [idx + offset for idx in batch["dataset_from_index"]],
                    "dataset_to_index": [idx + offset for idx in batch["dataset_to_index"]],
                },
                batched=True,
            )
            episode_offsets.append(offset_episodes)
            frame_offset += dataset.num_frames
        self.meta.episodes = datasets.concatenate_datasets(episode_offsets)
        self.set_image_transforms(image_transforms)

    def _get_camera_resize_shapes(self) -> dict[str, tuple[int, int, int]]:
        resize_shapes = {}
        camera_keys = set(self._datasets[0].meta.camera_keys)
        for dataset in self._datasets[1:]:
            camera_keys.intersection_update(dataset.meta.camera_keys)
        camera_keys.difference_update(self.disabled_features)

        for camera_key in sorted(camera_keys):
            shapes = [tuple(dataset.meta.features[camera_key]["shape"]) for dataset in self._datasets]
            channels = {shape[2] for shape in shapes}
            if len(channels) != 1:
                raise ValueError(f"Cannot combine camera '{camera_key}' with different channels: {shapes}.")
            target_height = max(shape[0] for shape in shapes)
            target_width = max(shape[1] for shape in shapes)
            resize_shapes[camera_key] = (channels.pop(), target_height, target_width)
            self.meta.info["features"][camera_key]["shape"] = (
                target_height,
                target_width,
                resize_shapes[camera_key][0],
            )
        return resize_shapes

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform for this dataset and its children."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(self.image_transforms)

    def clear_image_transforms(self) -> None:
        """Remove the transform from this dataset and its children."""
        self.set_image_transforms(None)

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update(
                {
                    k: v
                    for k, v in get_hf_features_from_features(dataset.features).items()
                    if k not in self.disabled_features
                }
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        for camera_key, target_shape in self._camera_resize_shapes.items():
            if camera_key in item:
                item[camera_key] = _resize_with_pad_image(item[camera_key], target_shape)
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
