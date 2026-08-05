#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
This script will help you convert any LeRobot dataset from codebase version 3.0 back to 2.1.
It will:

- Split consolidated data files back into per-episode parquet files
- Split consolidated video files back into per-episode video files
- Convert Parquet metadata back to JSONL format (episodes.jsonl, tasks.jsonl, episodes_stats.jsonl)
- Reconstruct nested stats structure from flattened columns
- Update codebase_version in `info.json` back to v2.1
- Optionally push this version to the hub on a specified branch and tag it with "v2.1"

Usage:

Convert a dataset from the hub:
```bash
python src/lerobot/scripts/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht
```

Convert a local dataset (works in place):
```bash
python src/lerobot/scripts/convert_dataset_v30_to_v21.py \
    --repo-id=lerobot/pusht \
    --root=/path/to/local/dataset/directory
```

"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from lerobot.utils.import_utils import require_package

require_package("jsonlines", extra="dataset")

import jsonlines
import pandas as pd
import tqdm
from datasets import Features, Image
from huggingface_hub import snapshot_download

from lerobot.datasets.io_utils import (
    load_info,
    load_json,
    write_info,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    INFO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    DatasetInfo,
)
from lerobot.datasets.video_utils import split_video_file
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging, unflatten_dict

V21 = "v2.1"
V30 = "v3.0"

"""
-------------------------
OLD (v3.0)
data/chunk-000/file_000.parquet

NEW (v2.1)
data/chunk-000/episode_000000.parquet
-------------------------
OLD (v3.0)
videos/CAMERA/chunk-000/file_000.mp4

NEW (v2.1)
videos/chunk-000/CAMERA/episode_000000.mp4
-------------------------
OLD (v3.0)
meta/episodes/chunk-000/file_000.parquet
episode_index | video_chunk_index | video_file_index | data_chunk_index | data_file_index | tasks | length

NEW (v2.1)
episodes.jsonl
{"episode_index": 1, "tasks": ["Put the blue block in the green bowl"], "length": 266}
-------------------------
OLD (v3.0)
meta/tasks.parquet
task_index | task

NEW (v2.1)
tasks.jsonl
{"task_index": 1, "task": "Put the blue block in the green bowl"}
-------------------------
OLD (v3.0)
meta/episodes/chunk-000/file_000.parquet (flattened stats columns)
episode_index | feature_name/min | feature_name/max | feature_name/mean | feature_name/std | feature_name/count

NEW (v2.1)
episodes_stats.jsonl
{"episode_index": 1, "stats": {"feature_name": {"min": ..., "max": ..., "mean": ..., "std": ..., "count": ...}}}
-------------------------
UPDATE
meta/info.json
-------------------------
"""


def write_jsonlines(data: list[dict], fpath: Path) -> None:
    """Write a list of dictionaries to a JSONL file."""
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def validate_local_dataset_version(local_path: Path) -> None:
    """Validate that the local dataset has the expected v3.0 version."""
    info = load_info(local_path)
    dataset_version = info.codebase_version or "unknown"
    if dataset_version != V30:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script is specifically for converting v3.0 datasets to v2.1."
        )


def load_episodes_metadata(root: Path) -> pd.DataFrame:
    """Load episodes metadata from v3.0 format (Parquet)."""
    episodes_dir = root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No episodes metadata found in {episodes_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def load_tasks_metadata(root: Path) -> pd.DataFrame:
    """Load tasks metadata from v3.0 format (Parquet)."""
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise ValueError(f"Tasks metadata not found at {tasks_path}")
    return pd.read_parquet(tasks_path)


def convert_tasks(root: Path, new_root: Path):
    """Convert tasks from Parquet back to JSONL format."""
    logging.info(f"Converting tasks from {root} to {new_root}")

    df_tasks = load_tasks_metadata(root)

    # Convert DataFrame to JSONL format
    tasks_list = []
    for task, row in df_tasks.iterrows():
        tasks_list.append({
            "task_index": int(row["task_index"]),
            "task": str(task)
        })

    write_jsonlines(tasks_list, new_root / LEGACY_TASKS_PATH)


def _to_python(value: Any) -> Any:
    """Recursively convert numpy / pyarrow scalars and arrays to plain Python types.

    Parquet columns backed by pyarrow can return numpy arrays, numpy scalars,
    or pyarrow arrays when read back via pandas.  ``json.dumps`` rejects all of
    these, so we normalise them here before writing JSONL.
    """
    import numpy as np

    if value is None:
        return None
    # numpy array → recurse on each element so nested arrays are also cleaned
    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    # numpy scalar (np.float32, np.int64, …) → Python scalar
    if isinstance(value, np.generic):
        return value.item()
    # pyarrow / pandas Extension arrays that expose tolist()
    if hasattr(value, "tolist"):
        result = value.tolist()
        # tolist() on a pyarrow array returns Python objects, but may still
        # contain numpy scalars if the backing type is numpy-derived.
        if isinstance(result, list):
            return [_to_python(v) for v in result]
        return _to_python(result)
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def extract_stats_from_flattened_columns(row: pd.Series) -> dict:
    """Extract and reconstruct nested stats structure from flattened column names.

    In v3.0, stats are stored as flattened columns named "stats/feature_name/metric"
    where values are numpy arrays (min/max/mean/std are vectors, count is shape (1,)).
    Feature names may themselves contain "/" so we strip the leading "stats/" prefix
    and split only on the *last* "/" to separate feature_name from metric.
    """
    stats = {}

    for col in row.index:
        if not col.startswith("stats/"):
            continue

        # Strip the leading "stats/" and split on the last "/" to get metric.
        # This correctly handles feature names that contain "/" themselves.
        remainder = col[len("stats/"):]
        if "/" not in remainder:
            continue
        feature_name, metric = remainder.rsplit("/", 1)

        if feature_name not in stats:
            stats[feature_name] = {}

        value = row[col]
        if value is None:
            continue
        stats[feature_name][metric] = _to_python(value)

    return stats


def convert_episodes_and_stats(root: Path, new_root: Path):
    """Convert episodes metadata and stats from Parquet back to JSONL format."""
    logging.info(f"Converting episodes and stats from {root} to {new_root}")

    df_episodes = load_episodes_metadata(root)

    episodes_list = []
    episodes_stats_list = []

    for _, row in df_episodes.iterrows():
        # Extract basic episode metadata
        episode_dict = {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"])
        }

        # Extract tasks. In v3.0 parquet this column comes back as a numpy
        # object array (e.g. array(['task description'], dtype=object)), not
        # a plain Python list, so normalise it via _to_python() first.
        if "tasks" in row:
            tasks = _to_python(row["tasks"])
            if isinstance(tasks, str):
                episode_dict["tasks"] = [tasks]
            elif isinstance(tasks, list):
                episode_dict["tasks"] = tasks
            else:
                episode_dict["tasks"] = []

        episodes_list.append(episode_dict)

        # Extract and reconstruct nested stats
        stats = extract_stats_from_flattened_columns(row)
        episodes_stats_list.append({
            "episode_index": int(row["episode_index"]),
            "stats": stats
        })

    write_jsonlines(episodes_list, new_root / LEGACY_EPISODES_PATH)
    write_jsonlines(episodes_stats_list, new_root / LEGACY_EPISODES_STATS_PATH)


def get_image_keys(root: Path) -> list[str]:
    """Get list of image feature keys from dataset info."""
    info = load_info(root)
    features = info.features
    return [key for key, ft in features.items() if ft["dtype"] == "image"]


def get_video_keys(root: Path) -> list[str]:
    """Get list of video feature keys from dataset info."""
    info = load_info(root)
    features = info.features
    return [key for key, ft in features.items() if ft["dtype"] == "video"]


def convert_data(root: Path, new_root: Path):
    """Split consolidated data files back into per-episode parquet files."""
    logging.info(f"Converting data files from {root} to {new_root}")

    df_episodes = load_episodes_metadata(root)
    image_keys = get_image_keys(root)

    # Group episodes by their data file location
    df_episodes["data_file"] = df_episodes.apply(
        lambda row: f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet",
        axis=1
    )

    grouped = df_episodes.groupby("data_file")

    for data_file, group in tqdm.tqdm(grouped, desc="Converting data files"):
        # Load the consolidated data file
        data_path = root / data_file
        if not data_path.exists():
            logging.warning(f"Data file not found: {data_path}")
            continue

        df_data = pd.read_parquet(data_path)

        # Compute per-file frame offset so iloc uses local (file-relative) indices.
        # dataset_from_index / dataset_to_index are global; subtract the smallest
        # from_index in this file to get the row position within df_data.
        file_frame_offset = int(group["dataset_from_index"].min())

        # Split into per-episode files
        for _, ep_row in group.iterrows():
            ep_idx = int(ep_row["episode_index"])
            from_idx = int(ep_row["dataset_from_index"]) - file_frame_offset
            to_idx = int(ep_row["dataset_to_index"]) - file_frame_offset

            # Extract episode data
            ep_data = df_data.iloc[from_idx:to_idx].reset_index(drop=True)

            # Determine chunk index for v2.1 structure
            chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE

            # Write to per-episode file
            output_path = new_root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle image features with proper schema
            if len(image_keys) > 0:
                schema = pd.io.parquet.schema.from_pandas(ep_data)
                features = Features.from_arrow_schema(schema)
                for key in image_keys:
                    if key in features:
                        features[key] = Image()
                schema = features.arrow_schema
            else:
                schema = None

            ep_data.to_parquet(output_path, index=False, schema=schema)


def convert_videos(root: Path, new_root: Path):
    """Split consolidated video files back into per-episode videos."""
    logging.info(f"Converting videos from {root} to {new_root}")

    video_keys = get_video_keys(root)
    if len(video_keys) == 0:
        logging.info("No videos to convert")
        return

    df_episodes = load_episodes_metadata(root)

    for video_key in tqdm.tqdm(video_keys, desc="Converting video cameras"):
        convert_videos_of_camera(root, new_root, video_key, df_episodes)


def convert_videos_of_camera(root: Path, new_root: Path, video_key: str, df_episodes: pd.DataFrame):
    """Split consolidated videos for a specific camera back into per-episode videos."""

    # Group episodes by their video file location for this camera
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_ts_col = f"videos/{video_key}/from_timestamp"
    to_ts_col = f"videos/{video_key}/to_timestamp"

    # Check if video columns exist
    if chunk_col not in df_episodes.columns:
        logging.warning(f"No video metadata found for {video_key}")
        return

    df_episodes["video_file"] = df_episodes.apply(
        lambda row: f"videos/{video_key}/chunk-{int(row[chunk_col]):03d}/file-{int(row[file_col]):03d}.mp4",
        axis=1
    )

    grouped = df_episodes.groupby("video_file")

    for video_file, group in tqdm.tqdm(grouped, desc=f"Converting {video_key} videos", leave=False):
        # Load the consolidated video file
        video_path = root / video_file
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            continue

        # Split into per-episode videos
        for _, ep_row in group.iterrows():
            ep_idx = int(ep_row["episode_index"])
            from_ts = float(ep_row[from_ts_col])
            to_ts = float(ep_row[to_ts_col])

            # Determine chunk index for v2.1 structure
            chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE

            # Write to per-episode video file
            output_path = new_root / f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{ep_idx:06d}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Split video using timestamps
            split_video_file(video_path, output_path, from_ts, to_ts)


def convert_info(root: Path, new_root: Path):
    """Update info.json from v3.0 back to v2.1 format."""
    logging.info(f"Converting info from {root} to {new_root}")

    # Load as raw dict to modify before constructing DatasetInfo
    info = load_json(root / INFO_PATH)

    # Update version
    info["codebase_version"] = V21

    # Remove v3.0 specific fields
    if "data_files_size_in_mb" in info:
        del info["data_files_size_in_mb"]
    if "video_files_size_in_mb" in info:
        del info["video_files_size_in_mb"]

    # Calculate total_chunks based on total_episodes
    total_episodes = info.get("total_episodes", 0)
    info["total_chunks"] = (total_episodes + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE

    # Set total_videos (equals total_episodes if videos exist)
    if info.get("video_path") is not None:
        info["total_videos"] = total_episodes
    else:
        info["total_videos"] = 0

    # Update data_path and video_path to v2.1 format.
    # lerobot v2.1 (0.3.3) formats these with episode_chunk / episode_index.
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    if info["video_path"] is not None:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    # v2.1 used integer fps, not float (0.3.3 codebase expects int).
    if "fps" in info:
        info["fps"] = int(info["fps"])

    # v2.1 video features had both "info" and "video_info" (duplicate) for
    # backward compatibility. v3.0 consolidated to just "info". Restore both.
    for key in info["features"]:
        if info["features"][key]["dtype"] == "video":
            if "info" in info["features"][key]:
                info["features"][key]["video_info"] = info["features"][key]["info"]
            # v3.0 already has fps in video_info, don't add it again
            continue
        # Non-video features in v2.1 didn't have fps
        if "fps" in info["features"][key]:
            del info["features"][key]["fps"]

    # Convert raw dict to typed DatasetInfo before writing
    dataset_info = DatasetInfo.from_dict(info)
    write_info(dataset_info, new_root)


def convert_dataset(
    repo_id: str,
    root: str | Path | None = None,
    force_conversion: bool = False,
):
    """
    Convert a dataset from v3.0 to v2.1 format (local only).

    Args:
        repo_id: Repository identifier used to locate the dataset under
            ``$HF_LEROBOT_HOME`` when ``root`` is not provided, and also to
            download the v3.0 snapshot from the Hub when a local copy does not
            yet exist.
        root: Local directory containing the v3.0 dataset. When omitted,
            defaults to ``$HF_LEROBOT_HOME/<repo_id>``. If the directory does
            not exist, the v3.0 snapshot is downloaded from the Hub first.
        force_conversion: Re-run conversion even if a local v2.1 copy is
            already present at ``root``.
    """
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root)

    old_root = root.parent / f"{root.name}_old"
    new_root = root.parent / f"{root.name}_v21"

    # If a previous interrupted run left both old_root and root, restore first.
    if old_root.is_dir() and root.is_dir():
        shutil.rmtree(str(root))
        shutil.move(str(old_root), str(root))

    if root.exists():
        if not force_conversion:
            info = load_info(root)
            if info.codebase_version == V21:
                print(f"Dataset at {root} is already v2.1. Use --force-conversion to convert anyway.")
                return
        validate_local_dataset_version(root)
        print(f"Using local dataset at {root}")
    else:
        print(f"Local dataset not found; downloading v3.0 snapshot from Hub…")
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=V30,
            local_dir=root,
        )

    if new_root.is_dir():
        shutil.rmtree(new_root)

    # Perform conversions
    convert_info(root, new_root)
    convert_tasks(root, new_root)
    convert_episodes_and_stats(root, new_root)
    convert_data(root, new_root)
    convert_videos(root, new_root)

    # Atomically swap: keep the v3.0 copy as <name>_old for safety
    shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))

    print(f"Conversion complete. v2.1 dataset written to: {root}")
    print(f"Original v3.0 dataset backed up to: {old_root}")


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `<USER>/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory containing the v3.0 dataset. Defaults to $HF_LEROBOT_HOME/repo_id.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force conversion even if the local dataset is already v2.1.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
