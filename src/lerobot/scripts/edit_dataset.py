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
"""Edit tool for LeRobot datasets (format **v2.1**).

This is the single entry point for all dataset editing operations. It currently
implements ``merge``, ``delete-episodes``, and ``rename-cameras`` for LeRobot
v2.1 datasets. Future operations (split, ...) will be added as new sub-commands
sharing the helpers in this module.

Scope / limitations (v1):
    - Only the on-disk **v2.1** format is supported (one file per episode,
      ``meta/episodes.jsonl`` / ``tasks.jsonl`` / ``episodes_stats.jsonl``).
      v3 / lerobot >= 0.4.0 datasets are not supported here; use the official
      ``lerobot-edit-dataset`` for those.
    - Video datasets and non-visual datasets are fully supported. Image datasets
      (dtype=="image", bytes embedded in parquet) are copied through best-effort
      but are not guaranteed.
    - Merge requires sources to be strictly compatible (same fps, features,
      chunks_size, video encoding info). Incompatible sources raise an error.

lerobot-edit-dataset rename-cameras \
  --input real_sim_dataset/rre_sortitem_yam_sim_v0 \
  --rename observation.images.first_person_camera_rgb=observation.images.head \
  --rename observation.images.left_hand_camera_rgb=observation.images.left_wrist \
  --rename observation.images.right_hand_camera_rgb=observation.images.right_wrist \
  --output real_sim_dataset/rre_sortitem_yam_sim_v0_renamed

Examples:

    Merge two local v2.1 datasets into one::

        lerobot-edit-dataset merge \\
            --inputs path/to/ds1 path/to/ds2 \\
            --output path/to/merged \\
            --repo-id user/merged_dataset

    Delete episodes 0 and 3 from a dataset (output to a new dir)::

        lerobot-edit-dataset delete-episodes \\
            --input path/to/ds \\
            --episode-indices 0 3 \\
            --output path/to/trimmed

    Delete episodes in place (original backed up to <ds>_backup_<ts>)::

        lerobot-edit-dataset delete-episodes \\
            --input path/to/ds \\
            --episode-indices 0 3 \\
            --in-place

    Rename cameras (short names or full feature keys)::

        lerobot-edit-dataset rename-cameras \\
            --input path/to/ds \\
            --rename left_hand_camera_rgb=cam_left \\
            --rename observation.images.right_hand_camera_rgb=observation.images.cam_right \\
            --output path/to/renamed
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import packaging.version
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.constants import OBS_IMAGES
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    cast_stats_to_numpy,
    load_episodes,
    load_episodes_stats,
    load_json,
    load_tasks,
    load_jsonlines,
    serialize_dict,
    write_json,
    write_jsonlines,
    write_stats,
)

logger = logging.getLogger(__name__)

# Fields of a video feature's global "info" block that must match across sources
# (see lerobot.datasets.video_utils.get_video_info). The block is the same for
# every episode of a given video key, read once from episode 0.
_VIDEO_INFO_FIELDS = [
    "video.height",
    "video.width",
    "video.codec",
    "video.pix_fmt",
    "video.fps",
    "video.channels",
]


# =====================================================================================
# Shared helpers (reusable by future sub-commands)
# =====================================================================================
def load_meta(root: str | Path, repo_id: str | None = None) -> LeRobotDatasetMetadata:
    """Load the (read-only) metadata of a local v2.1 dataset.

    Loads purely from disk; does not hit the network as long as ``meta/`` is
    present locally (which it must be for an edit operation).
    """
    root = Path(root).expanduser()
    if not (root / INFO_PATH).is_file():
        raise FileNotFoundError(f"No dataset metadata found at {root / INFO_PATH}")
    repo_id = repo_id if repo_id is not None else root.name
    # force_cache_sync=False + present meta => offline load.
    return LeRobotDatasetMetadata(repo_id=repo_id, root=root, force_cache_sync=False)


def _video_info(ft: dict) -> dict | None:
    return ft.get("info") if ft.get("dtype") == "video" else None


def _diff_features(feat_a: dict, feat_b: dict) -> list[str]:
    """Return human-readable differences between two ``info["features"]`` dicts."""
    diffs: list[str] = []
    keys_a, keys_b = set(feat_a), set(feat_b)
    if keys_a != keys_b:
        diffs.append(
            f"feature key sets differ: only in first={sorted(keys_a - keys_b)}, "
            f"only in second={sorted(keys_b - keys_a)}"
        )
    for key in sorted(keys_a & keys_b):
        a, b = feat_a[key], feat_b[key]
        if a.get("dtype") != b.get("dtype"):
            diffs.append(f"feature '{key}': dtype {a.get('dtype')!r} vs {b.get('dtype')!r}")
            continue
        if tuple(a.get("shape", ())) != tuple(b.get("shape", ())):
            diffs.append(f"feature '{key}': shape {a.get('shape')} vs {b.get('shape')}")
        if a.get("names") != b.get("names"):
            diffs.append(f"feature '{key}': names {a.get('names')} vs {b.get('names')}")
        if a.get("dtype") == "video":
            ia, ib = _video_info(a) or {}, _video_info(b) or {}
            for fld in _VIDEO_INFO_FIELDS:
                if ia.get(fld) != ib.get(fld):
                    diffs.append(f"video feature '{key}': {fld} {ia.get(fld)!r} vs {ib.get(fld)!r}")
    return diffs


def validate_compatible(metas: list[LeRobotDatasetMetadata]) -> None:
    """Strict compatibility check across source dataset metadatas.

    Raises ValueError with the full list of incompatibilities if any source
    differs from the first one.
    """
    if len(metas) < 2:
        return
    base = metas[0]
    problems: list[str] = []

    # codebase_version must be v2.1 (we only operate on v2.1 on-disk layout).
    for m in metas:
        v = packaging.version.parse(m.info["codebase_version"])
        if v != packaging.version.parse(CODEBASE_VERSION):
            problems.append(
                f"dataset '{m.repo_id}' has codebase_version {m.info['codebase_version']}; "
                f"this tool only supports {CODEBASE_VERSION}."
            )

    base_fps = base.info["fps"]
    base_chunks = base.info["chunks_size"]
    base_features = base.info["features"]
    base_robot = base.info.get("robot_type")

    for m in metas[1:]:
        tag = f"[{m.repo_id}]"
        if m.info["fps"] != base_fps:
            problems.append(f"{tag} fps {m.info['fps']} != {base_fps}")
        if m.info["chunks_size"] != base_chunks:
            problems.append(f"{tag} chunks_size {m.info['chunks_size']} != {base_chunks}")
        if m.info.get("robot_type") != base_robot:
            problems.append(
                f"{tag} robot_type {m.info.get('robot_type')!r} != {base_robot!r}"
            )
        problems.extend(f"{tag} {d}" for d in _diff_features(base_features, m.info["features"]))

    if problems:
        msg = "Source datasets are not compatible for merging:\n  - " + "\n  - ".join(problems)
        raise ValueError(msg)


def _episode_chunk(ep_index: int, chunks_size: int) -> int:
    return ep_index // chunks_size


def _format_parquet_path(ep_index: int, chunks_size: int, template: str) -> Path:
    return Path(
        template.format(episode_chunk=_episode_chunk(ep_index, chunks_size), episode_index=ep_index)
    )


def _format_video_path(ep_index: int, chunks_size: int, template: str, video_key: str) -> Path:
    return Path(
        template.format(
            episode_chunk=_episode_chunk(ep_index, chunks_size),
            video_key=video_key,
            episode_index=ep_index,
        )
    )


def _rewrite_episode_parquet(
    src_path: Path,
    dst_path: Path,
    new_episode_index: int,
    new_index_start: int,
    task_index_map: dict[int, int] | None = None,
) -> int:
    """Rewrite a single episode parquet, remapping ``index``/``episode_index``/``task_index``.

    ``task_index_map`` remaps source task_index -> new task_index. If ``None``, the
    ``task_index`` column is kept unchanged (used by delete_episodes, where the task
    table is preserved as-is).

    Returns the episode length (number of frames).
    """
    table = pq.read_table(src_path)
    length = table.num_rows

    new_index = pa.array(np.arange(new_index_start, new_index_start + length, dtype=np.int64))
    new_ep = pa.array(np.full(length, new_episode_index, dtype=np.int64))

    columns = []
    for name in table.schema.names:
        if name == "index":
            columns.append(new_index)
        elif name == "episode_index":
            columns.append(new_ep)
        elif name == "task_index":
            if task_index_map is None:
                columns.append(table.column(name))
            else:
                old_col = table.column(name).to_pylist()
                new_col = pa.array(
                    np.array([task_index_map[int(t)] for t in old_col], dtype=np.int64)
                )
                columns.append(new_col)
        else:
            columns.append(table.column(name))

    new_table = pa.table(columns, schema=table.schema)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return length


# =====================================================================================
# merge operation
# =====================================================================================
def merge_datasets(
    inputs: list[str | Path],
    output: str | Path,
    repo_id: str | None = None,
) -> Path:
    """Merge several LeRobot v2.1 datasets into a new one at ``output``.

    Args:
        inputs: Local root directories of the source datasets.
        output: Destination directory (must not exist or be empty).
        repo_id: repo_id to record in the output info.json (defaults to the
            output directory name).

    Returns:
        The output dataset root path.
    """
    if len(inputs) < 2:
        raise ValueError("merge requires at least two input datasets.")
    input_roots = [Path(p).expanduser().resolve() for p in inputs]
    for p in input_roots:
        if not (p / INFO_PATH).is_file():
            raise FileNotFoundError(f"Input is not a LeRobot dataset (missing {INFO_PATH}): {p}")

    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory {output} already exists and is not empty. "
            "Remove it or choose a different --output."
        )
    for p in input_roots:
        if p == output:
            raise ValueError("Output path must not equal any input path.")

    repo_id = repo_id if repo_id is not None else output.name
    metas = [load_meta(p) for p in input_roots]
    validate_compatible(metas)

    base_meta = metas[0]
    chunks_size = base_meta.chunks_size
    data_path_template = base_meta.info["data_path"]
    video_path_template = base_meta.info.get("video_path")
    video_keys = base_meta.video_keys
    has_image = len(base_meta.image_keys) > 0
    if has_image:
        logger.warning(
            "Dataset contains image (embedded-bytes) features. They are copied through "
            "best-effort; correctness is not guaranteed for this v1."
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    # Output info: clone the base, reset all counters, rebuild metadata files.
    out_info = copy.deepcopy(base_meta.info)
    out_info["repo_id"] = repo_id
    out_info["total_episodes"] = 0
    out_info["total_frames"] = 0
    out_info["total_tasks"] = 0
    out_info["total_videos"] = 0
    out_info["total_chunks"] = 0
    out_info["splits"] = {}
    # video info blocks already cloned from base (episode-0-derived global info).

    # Running state.
    next_ep_index = 0
    next_global_frame = 0
    # task_str -> merged task_index
    task_table: dict[str, int] = {}
    all_episode_stats: list[dict] = []  # for global stats aggregation

    episodes_out_path = output / EPISODES_PATH
    episodes_stats_out_path = output / EPISODES_STATS_PATH
    tasks_out_path = output / TASKS_PATH

    n_video_keys = len(video_keys)

    # Pre-load per-source episodes (used for the progress bar total and the loop).
    per_src_episodes = [load_episodes(p) for p in input_roots]
    per_src_episodes_stats = [load_episodes_stats(p) for p in input_roots]
    total_episodes_all = sum(len(e) for e in per_src_episodes)
    logger.info(
        "Merging %d source dataset(s): %d episode(s), %d video key(s).",
        len(input_roots), total_episodes_all, n_video_keys,
    )

    pbar = tqdm(
        total=total_episodes_all,
        unit="ep",
        desc="merge",
        dynamic_ncols=True,
    )
    try:
        for src_root, meta, src_episodes, src_episodes_stats in zip(
            input_roots, metas, per_src_episodes, per_src_episodes_stats, strict=True
        ):
            # Build per-source task_index remap: source task_index -> task string -> merged task_index.
            # IMPORTANT: use meta.tasks ({idx: str}) not task_to_task_index reversed, because a source may
            # have multiple task_index values pointing to the SAME task string (real datasets do this).
            # meta.tasks preserves every source task_index; we then dedup by string into the merged table,
            # so duplicate-string source indices collapse onto one merged index.
            src_idx_to_str = dict(meta.tasks)  # {src_task_index: task_str}
            task_idx_remap: dict[int, int] = {}

            for src_ep_idx in sorted(src_episodes.keys()):
                ep_dict = src_episodes[src_ep_idx]
                length = ep_dict["length"]
                ep_task_strs = ep_dict.get("tasks", [])

                # Remap task strings into the merged task table, appending new tasks.
                for t in ep_task_strs:
                    if t not in task_table:
                        merged_idx = out_info["total_tasks"]
                        task_table[t] = merged_idx
                        out_info["total_tasks"] += 1
                        append_jsonlines({"task_index": merged_idx, "task": t}, tasks_out_path)

                # Build src task_index -> merged task_index map lazily (all source task indices).
                for src_idx, t in src_idx_to_str.items():
                    if src_idx not in task_idx_remap:
                        if t in task_table:
                            task_idx_remap[src_idx] = task_table[t]
                        else:
                            # A task index appears in tasks.jsonl but not in any episode; still remap.
                            merged_idx = out_info["total_tasks"]
                            task_table[t] = merged_idx
                            out_info["total_tasks"] += 1
                            append_jsonlines({"task_index": merged_idx, "task": t}, tasks_out_path)
                            task_idx_remap[src_idx] = merged_idx

                # 1. Rewrite parquet.
                src_pq = src_root / _format_parquet_path(
                    src_ep_idx, chunks_size, data_path_template
                )
                dst_pq = output / _format_parquet_path(
                    next_ep_index, chunks_size, data_path_template
                )
                actual_length = _rewrite_episode_parquet(
                    src_pq,
                    dst_pq,
                    new_episode_index=next_ep_index,
                    new_index_start=next_global_frame,
                    task_index_map=task_idx_remap,
                )
                if actual_length != length:
                    logger.warning(
                        f"[{meta.repo_id}] episode {src_ep_idx}: meta length {length} != "
                        f"parquet rows {actual_length}; using parquet rows."
                    )
                    length = actual_length

                # 2. Copy videos (raw bytes, no re-encode).
                if video_path_template is not None:
                    for vid_key in video_keys:
                        src_vid = src_root / _format_video_path(
                            src_ep_idx, chunks_size, video_path_template, vid_key
                        )
                        if not src_vid.is_file():
                            raise FileNotFoundError(
                                f"Video file missing for [{meta.repo_id}] episode {src_ep_idx} "
                                f"key {vid_key}: {src_vid}"
                            )
                        dst_vid = output / _format_video_path(
                            next_ep_index, chunks_size, video_path_template, vid_key
                        )
                        dst_vid.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_vid, dst_vid)

                # 3. episodes.jsonl entry.
                append_jsonlines(
                    {
                        "episode_index": next_ep_index,
                        "tasks": list(ep_task_strs),
                        "length": length,
                    },
                    episodes_out_path,
                )

                # 4. episodes_stats.jsonl entry (remap episode_index only; stats unchanged).
                if src_ep_idx in src_episodes_stats:
                    ep_stats = src_episodes_stats[src_ep_idx]
                    stats_row = {
                        "episode_index": next_ep_index,
                        "stats": serialize_dict(
                            {k: v for k, v in ep_stats.items()}
                        ),
                    }
                    append_jsonlines(stats_row, episodes_stats_out_path)
                    all_episode_stats.append(ep_stats)

                # 5. Update running counters.
                next_ep_index += 1
                next_global_frame += length
                out_info["total_episodes"] += 1
                out_info["total_frames"] += length
                out_info["total_videos"] += n_video_keys
                chunk_now = _episode_chunk(next_ep_index - 1, chunks_size)
                if chunk_now + 1 > out_info["total_chunks"]:
                    out_info["total_chunks"] = chunk_now + 1
                out_info["splits"] = {"train": f"0:{out_info['total_episodes']}"}

                pbar.set_postfix(
                    src=meta.repo_id,
                    frames=out_info["total_frames"],
                    videos=out_info["total_videos"],
                )
                pbar.update(1)
    finally:
        pbar.close()

    if next_ep_index == 0:
        raise ValueError("No episodes found across input datasets.")

    # Write info.json.
    write_json(out_info, output / INFO_PATH)

    # Write global stats.json from aggregated per-episode stats.
    if all_episode_stats:
        global_stats = aggregate_stats(all_episode_stats)
        write_stats(global_stats, output)
    else:
        logger.warning("No per-episode stats found; skipping stats.json.")

    _consistency_check(output, out_info, video_keys, video_path_template)

    logger.info(
        "Merge complete: %d source(s) -> %d episodes, %d frames, %d task(s), %d video(s). "
        "Output: %s",
        len(input_roots),
        out_info["total_episodes"],
        out_info["total_frames"],
        out_info["total_tasks"],
        out_info["total_videos"],
        output,
    )
    return output


# =====================================================================================
# delete_episodes operation
# =====================================================================================
def delete_episodes(
    input: str | Path,
    episode_indices: list[int],
    output: str | Path | None = None,
    repo_id: str | None = None,
    in_place: bool = False,
) -> Path:
    """Delete episodes from a LeRobot v2.1 dataset, reindexing the survivors 0..N-1.

    Behaviour:
        - Episodes listed in ``episode_indices`` are removed.
        - Surviving episodes are re-indexed contiguously from 0 (required by the v2.1
          format, which derives chunk/parquet/mp4 paths from ``episode_index``).
        - Per-frame ``index`` (global frame counter) and ``episode_index`` columns are
          rewritten; ``task_index`` is kept as-is (the task table is preserved, so
          indices remain valid strings regardless).
        - ``meta/episodes.jsonl`` / ``episodes_stats.jsonl`` are rebuilt; ``stats.json``
          is re-aggregated; ``info.json`` counters/splits updated.

    Args:
        input: Source dataset root directory.
        episode_indices: Source episode_index values to delete.
        output: Destination directory (required unless ``in_place``). Must be empty/absent.
        repo_id: repo_id recorded in the output info.json (default: input's repo_id).
        in_place: If True, write back into ``input`` (after copying the original aside as
            a ``<input>_backup_<ts>`` sibling so the operation is reversible). Exclusive
            with ``output``.

    Returns:
        The output dataset root path.
    """
    import time

    src = Path(input).expanduser().resolve()
    if not (src / INFO_PATH).is_file():
        raise FileNotFoundError(f"Input is not a LeRobot dataset (missing {INFO_PATH}): {src}")

    if in_place and output is not None:
        raise ValueError("Cannot specify both --output and --in-place.")
    if not in_place and output is None:
        raise ValueError("Either --output or --in-place must be specified.")

    meta = load_meta(src)
    v = packaging.version.parse(meta.info["codebase_version"])
    if v != packaging.version.parse(CODEBASE_VERSION):
        raise ValueError(
            f"delete_episodes only supports {CODEBASE_VERSION}; got {meta.info['codebase_version']}."
        )

    del_set = set(int(i) for i in episode_indices)
    all_eps = sorted(meta.episodes.keys())
    for idx in del_set:
        if idx not in meta.episodes:
            raise ValueError(
                f"Episode index {idx} not found in dataset '{meta.repo_id}' "
                f"(valid: 0..{max(all_eps) if all_eps else -1})."
            )
    keep_eps = [e for e in all_eps if e not in del_set]
    n_total = len(all_eps)
    n_keep = len(keep_eps)
    if n_keep == 0:
        raise ValueError("Refusing to delete every episode (result would be empty).")

    # Resolve output dir.
    if in_place:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = src.parent / f"{src.name}_backup_{ts}"
        if backup.exists():
            raise FileExistsError(f"Backup path already exists: {backup}")
        output_root = src
        # Move original aside, then we will rebuild in_place into a fresh dir and finally
        # swap. Simplest reversible approach: rename src -> backup, then treat backup as input.
        shutil.move(str(src), str(backup))
        source_dir = backup
        logger.info("Moved original dataset to backup: %s", backup)
    else:
        output_root = Path(output).expanduser().resolve()  # type: ignore[arg-type]
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"Output directory {output_root} already exists and is not empty."
            )
        if output_root == src:
            raise ValueError("Output path must not equal input path (use --in-place instead).")
        source_dir = src

    chunks_size = meta.chunks_size
    data_path_template = meta.info["data_path"]
    video_path_template = meta.info.get("video_path")
    video_keys = meta.video_keys

    out_info = copy.deepcopy(meta.info)
    out_info["repo_id"] = repo_id if repo_id is not None else meta.repo_id
    out_info["total_episodes"] = 0
    out_info["total_frames"] = 0
    out_info["total_videos"] = 0
    out_info["total_chunks"] = 0
    out_info["splits"] = {}
    # total_tasks unchanged: task table preserved as-is.

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    # Preserve the task table verbatim.
    src_tasks_path = source_dir / TASKS_PATH
    if src_tasks_path.is_file():
        shutil.copy2(src_tasks_path, output_root / TASKS_PATH)

    episodes_out_path = output_root / EPISODES_PATH
    episodes_stats_out_path = output_root / EPISODES_STATS_PATH

    src_episodes_stats = load_episodes_stats(source_dir)
    all_episode_stats: list[dict] = []

    next_ep_index = 0
    next_global_frame = 0

    logger.info(
        "Deleting %d episode(s) from '%s': %d -> %d remaining.",
        len(del_set), meta.repo_id, n_total, n_keep,
    )

    pbar = tqdm(total=n_keep, unit="ep", desc="delete", dynamic_ncols=True)
    try:
        for src_ep_idx in keep_eps:
            ep_dict = meta.episodes[src_ep_idx]
            length = ep_dict["length"]
            ep_task_strs = ep_dict.get("tasks", [])

            # 1. Rewrite parquet (episode_index + index; task_index unchanged).
            src_pq = source_dir / _format_parquet_path(src_ep_idx, chunks_size, data_path_template)
            if not src_pq.is_file():
                raise FileNotFoundError(f"Missing parquet for episode {src_ep_idx}: {src_pq}")
            dst_pq = output_root / _format_parquet_path(next_ep_index, chunks_size, data_path_template)
            actual_length = _rewrite_episode_parquet(
                src_pq, dst_pq,
                new_episode_index=next_ep_index,
                new_index_start=next_global_frame,
                task_index_map=None,
            )
            if actual_length != length:
                logger.warning(
                    "episode %d: meta length %d != parquet rows %d; using parquet rows.",
                    src_ep_idx, length, actual_length,
                )
                length = actual_length

            # 2. Copy videos.
            if video_path_template is not None:
                for vid_key in video_keys:
                    src_vid = source_dir / _format_video_path(
                        src_ep_idx, chunks_size, video_path_template, vid_key
                    )
                    if not src_vid.is_file():
                        raise FileNotFoundError(
                            f"Video missing for episode {src_ep_idx} key {vid_key}: {src_vid}"
                        )
                    dst_vid = output_root / _format_video_path(
                        next_ep_index, chunks_size, video_path_template, vid_key
                    )
                    dst_vid.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_vid, dst_vid)

            # 3. episodes.jsonl
            append_jsonlines(
                {"episode_index": next_ep_index, "tasks": list(ep_task_strs), "length": length},
                episodes_out_path,
            )

            # 4. episodes_stats.jsonl
            if src_ep_idx in src_episodes_stats:
                ep_stats = src_episodes_stats[src_ep_idx]
                append_jsonlines(
                    {"episode_index": next_ep_index, "stats": serialize_dict(ep_stats)},
                    episodes_stats_out_path,
                )
                all_episode_stats.append(ep_stats)

            # 5. counters
            next_ep_index += 1
            next_global_frame += length
            out_info["total_episodes"] += 1
            out_info["total_frames"] += length
            out_info["total_videos"] += len(video_keys)
            chunk_now = _episode_chunk(next_ep_index - 1, chunks_size)
            if chunk_now + 1 > out_info["total_chunks"]:
                out_info["total_chunks"] = chunk_now + 1
            out_info["splits"] = {"train": f"0:{out_info['total_episodes']}"}

            pbar.set_postfix(frames=out_info["total_frames"])
            pbar.update(1)
    finally:
        pbar.close()

    write_json(out_info, output_root / INFO_PATH)

    if all_episode_stats:
        write_stats(aggregate_stats(all_episode_stats), output_root)
    else:
        logger.warning("No per-episode stats found; skipping stats.json.")

    _consistency_check(output_root, out_info, video_keys, video_path_template)

    logger.info(
        "Delete complete: '%s' kept %d episode(s), %d frame(s), %d video(s). Output: %s",
        meta.repo_id, out_info["total_episodes"], out_info["total_frames"],
        out_info["total_videos"], output_root,
    )
    return output_root


def _consistency_check(
    root: Path, info: dict, video_keys: list[str], video_path_template: str | None
) -> None:
    """Verify file/metadata counts match after writing."""
    parquet_files = list(root.rglob("*.parquet"))
    if len(parquet_files) != info["total_episodes"]:
        raise RuntimeError(
            f"Consistency check failed: {len(parquet_files)} parquet files != "
            f"total_episodes {info['total_episodes']}"
        )

    if video_path_template is not None and video_keys:
        video_files = list(root.rglob("*.mp4"))
        expected = info["total_episodes"] * len(video_keys)
        if len(video_files) != expected:
            raise RuntimeError(
                f"Consistency check failed: {len(video_files)} mp4 files != "
                f"expected {expected} ({info['total_episodes']} episodes x {len(video_keys)} keys)"
            )

    episodes = load_jsonlines(root / EPISODES_PATH)
    if len(episodes) != info["total_episodes"]:
        raise RuntimeError(
            f"Consistency check failed: {len(episodes)} episodes.jsonl lines != "
            f"total_episodes {info['total_episodes']}"
        )
    ep_stats = load_jsonlines(root / EPISODES_STATS_PATH)
    if len(ep_stats) != info["total_episodes"]:
        raise RuntimeError(
            f"Consistency check failed: {len(ep_stats)} episodes_stats.jsonl lines != "
            f"total_episodes {info['total_episodes']}"
        )
    tasks = load_jsonlines(root / TASKS_PATH)
    if len(tasks) != info["total_tasks"]:
        raise RuntimeError(
            f"Consistency check failed: {len(tasks)} tasks.jsonl lines != "
            f"total_tasks {info['total_tasks']}"
        )
    logger.info("Consistency check passed.")


# =====================================================================================
# rename_cameras operation
# =====================================================================================
def _normalize_camera_key(name: str) -> str:
    """Accept a full feature key or a short camera name.

    Short names (no ``.``) are prefixed with ``observation.images.``.
    """
    name = name.strip()
    if not name:
        raise ValueError("Camera key must be non-empty.")
    if "." in name:
        return name
    return f"{OBS_IMAGES}.{name}"


def _parse_rename_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse ``OLD=NEW`` CLI pairs into a normalized old->new mapping."""
    mapping: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid --rename value {raw!r}; expected OLD=NEW.")
        old_raw, new_raw = raw.split("=", 1)
        old_key = _normalize_camera_key(old_raw)
        new_key = _normalize_camera_key(new_raw)
        if old_key in mapping and mapping[old_key] != new_key:
            raise ValueError(f"Duplicate rename source {old_key!r}: {mapping[old_key]!r} vs {new_key!r}.")
        mapping[old_key] = new_key
    # Drop no-ops.
    return {old: new for old, new in mapping.items() if old != new}


def _validate_camera_rename_map(
    features: dict,
    rename_map: dict[str, str],
) -> None:
    """Ensure ``rename_map`` only remaps existing cameras without key collisions."""
    if not rename_map:
        raise ValueError("No effective renames provided (empty map or all OLD==NEW).")

    camera_keys = {k for k, ft in features.items() if ft.get("dtype") in ("video", "image")}
    all_keys = set(features)

    for old in rename_map:
        if old not in camera_keys:
            raise ValueError(
                f"Cannot rename {old!r}: not an existing camera key. "
                f"Available cameras: {sorted(camera_keys)}"
            )

    new_targets = list(rename_map.values())
    if len(new_targets) != len(set(new_targets)):
        raise ValueError(f"Duplicate rename targets in map: {rename_map}")

    # Simulate final key set for collision detection.
    final_keys: set[str] = set()
    for key in all_keys:
        final = rename_map.get(key, key)
        if final in final_keys:
            raise ValueError(
                f"Rename collision: multiple features would become {final!r}. Map={rename_map}"
            )
        final_keys.add(final)

    # New camera names must not collide with non-camera features that stay put.
    for old, new in rename_map.items():
        if new in all_keys and new not in rename_map and new != old:
            # `new` already exists and is not itself being renamed away.
            raise ValueError(
                f"Cannot rename {old!r} -> {new!r}: target key already exists in features."
            )


def _rename_dict_keys(d: dict, rename_map: dict[str, str]) -> dict:
    """Return a shallow-key-renamed copy of ``d`` (values reused)."""
    out: dict = {}
    for key, value in d.items():
        new_key = rename_map.get(key, key)
        if new_key in out:
            raise ValueError(f"Key collision while renaming stats/features at {new_key!r}.")
        out[new_key] = value
    return out


def _copy_or_rename_parquet_columns(
    src_path: Path,
    dst_path: Path,
    rename_map: dict[str, str],
) -> None:
    """Copy a parquet file, renaming columns that appear in ``rename_map``."""
    table = pq.read_table(src_path)
    old_names = list(table.schema.names)
    new_names = [rename_map.get(n, n) for n in old_names]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if new_names == old_names:
        shutil.copy2(src_path, dst_path)
        return
    if len(new_names) != len(set(new_names)):
        raise ValueError(f"Parquet column rename collision in {src_path}: {old_names} -> {new_names}")
    pq.write_table(table.rename_columns(new_names), dst_path)


def rename_cameras(
    input: str | Path,
    rename: list[str] | dict[str, str],
    output: str | Path | None = None,
    repo_id: str | None = None,
    in_place: bool = False,
) -> Path:
    """Rename camera feature keys in a LeRobot v2.1 dataset.

    Updates ``info.json`` features, ``stats.json``, ``episodes_stats.jsonl``,
    video directory names (``videos/.../{video_key}/...``), and parquet column
    names when image cameras are embedded as columns.

    Args:
        input: Source dataset root directory.
        rename: Either a list of ``OLD=NEW`` strings or an already-parsed
            ``{old_key: new_key}`` mapping. Short names (no ``.``) are prefixed
            with ``observation.images.``.
        output: Destination directory (required unless ``in_place``). Must be empty/absent.
        repo_id: repo_id recorded in the output info.json (default: keep source).
        in_place: If True, write back into ``input`` after moving the original aside
            as ``<input>_backup_<ts>``. Exclusive with ``output``.

    Returns:
        The output dataset root path.
    """
    import time

    src = Path(input).expanduser().resolve()
    if not (src / INFO_PATH).is_file():
        raise FileNotFoundError(f"Input is not a LeRobot dataset (missing {INFO_PATH}): {src}")

    if in_place and output is not None:
        raise ValueError("Cannot specify both --output and --in-place.")
    if not in_place and output is None:
        raise ValueError("Either --output or --in-place must be specified.")

    meta = load_meta(src)
    v = packaging.version.parse(meta.info["codebase_version"])
    if v != packaging.version.parse(CODEBASE_VERSION):
        raise ValueError(
            f"rename_cameras only supports {CODEBASE_VERSION}; got {meta.info['codebase_version']}."
        )

    if isinstance(rename, dict):
        rename_map = {
            _normalize_camera_key(k): _normalize_camera_key(v)
            for k, v in rename.items()
            if _normalize_camera_key(k) != _normalize_camera_key(v)
        }
    else:
        rename_map = _parse_rename_pairs(list(rename))
    _validate_camera_rename_map(meta.info["features"], rename_map)

    # Resolve output dir.
    if in_place:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = src.parent / f"{src.name}_backup_{ts}"
        if backup.exists():
            raise FileExistsError(f"Backup path already exists: {backup}")
        output_root = src
        shutil.move(str(src), str(backup))
        source_dir = backup
        logger.info("Moved original dataset to backup: %s", backup)
    else:
        output_root = Path(output).expanduser().resolve()  # type: ignore[arg-type]
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"Output directory {output_root} already exists and is not empty."
            )
        if output_root == src:
            raise ValueError("Output path must not equal input path (use --in-place instead).")
        source_dir = src

    chunks_size = meta.chunks_size
    data_path_template = meta.info["data_path"]
    video_path_template = meta.info.get("video_path")
    src_video_keys = list(meta.video_keys)
    src_image_keys = list(meta.image_keys)
    needs_parquet_rewrite = any(k in rename_map for k in src_image_keys)

    out_info = copy.deepcopy(meta.info)
    out_info["features"] = _rename_dict_keys(out_info["features"], rename_map)
    out_info["repo_id"] = repo_id if repo_id is not None else meta.repo_id
    out_video_keys = [rename_map.get(k, k) for k in src_video_keys]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    # Preserve task + episode tables verbatim.
    for rel in (TASKS_PATH, EPISODES_PATH):
        src_path = source_dir / rel
        if src_path.is_file():
            shutil.copy2(src_path, output_root / rel)

    # Rewrite stats with renamed camera keys (no recompute).
    src_stats_path = source_dir / STATS_PATH
    if src_stats_path.is_file():
        stats = load_json(src_stats_path)
        write_json(_rename_dict_keys(stats, rename_map), output_root / STATS_PATH)
    else:
        logger.warning("No stats.json found; skipping.")

    src_ep_stats_path = source_dir / EPISODES_STATS_PATH
    if src_ep_stats_path.is_file():
        ep_stats_lines = load_jsonlines(src_ep_stats_path)
        rewritten = []
        for row in ep_stats_lines:
            row = dict(row)
            if "stats" in row and isinstance(row["stats"], dict):
                row["stats"] = _rename_dict_keys(row["stats"], rename_map)
            rewritten.append(row)
        write_jsonlines(rewritten, output_root / EPISODES_STATS_PATH)
    else:
        logger.warning("No episodes_stats.jsonl found; skipping.")

    all_eps = sorted(meta.episodes.keys())
    logger.info(
        "Renaming %d camera(s) in '%s' (%d episodes): %s",
        len(rename_map),
        meta.repo_id,
        len(all_eps),
        ", ".join(f"{o} -> {n}" for o, n in rename_map.items()),
    )

    pbar = tqdm(total=len(all_eps), unit="ep", desc="rename-cameras", dynamic_ncols=True)
    try:
        for ep_idx in all_eps:
            # 1. Parquet: rewrite columns only when image cameras are renamed.
            src_pq = source_dir / _format_parquet_path(ep_idx, chunks_size, data_path_template)
            if not src_pq.is_file():
                raise FileNotFoundError(f"Missing parquet for episode {ep_idx}: {src_pq}")
            dst_pq = output_root / _format_parquet_path(ep_idx, chunks_size, data_path_template)
            if needs_parquet_rewrite:
                _copy_or_rename_parquet_columns(src_pq, dst_pq, rename_map)
            else:
                dst_pq.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_pq, dst_pq)

            # 2. Videos: copy under possibly-new video_key directories.
            if video_path_template is not None:
                for src_key in src_video_keys:
                    dst_key = rename_map.get(src_key, src_key)
                    src_vid = source_dir / _format_video_path(
                        ep_idx, chunks_size, video_path_template, src_key
                    )
                    if not src_vid.is_file():
                        raise FileNotFoundError(
                            f"Video missing for episode {ep_idx} key {src_key}: {src_vid}"
                        )
                    dst_vid = output_root / _format_video_path(
                        ep_idx, chunks_size, video_path_template, dst_key
                    )
                    dst_vid.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_vid, dst_vid)

            # 3. Optional standalone image directories (best-effort).
            for src_key in src_image_keys:
                dst_key = rename_map.get(src_key, src_key)
                src_img_dir = source_dir / "images" / src_key
                if not src_img_dir.is_dir():
                    continue
                dst_img_dir = output_root / "images" / dst_key
                # Copy only this episode's frames if present; else copy whole tree once.
                ep_src = src_img_dir / f"episode_{ep_idx:06d}"
                if ep_src.is_dir():
                    dst_ep = dst_img_dir / f"episode_{ep_idx:06d}"
                    if dst_ep.exists():
                        shutil.rmtree(dst_ep)
                    shutil.copytree(ep_src, dst_ep)

            pbar.update(1)
    finally:
        pbar.close()

    # Best-effort: if images/ exists but was not episode-structured, copy remaining dirs.
    src_images_root = source_dir / "images"
    if src_images_root.is_dir():
        for src_key in src_image_keys:
            dst_key = rename_map.get(src_key, src_key)
            src_img_dir = src_images_root / src_key
            dst_img_dir = output_root / "images" / dst_key
            if src_img_dir.is_dir() and not dst_img_dir.exists():
                shutil.copytree(src_img_dir, dst_img_dir)

    write_json(out_info, output_root / INFO_PATH)
    _consistency_check(output_root, out_info, out_video_keys, video_path_template)

    logger.info(
        "Rename complete: '%s' -> %d camera rename(s). Output: %s",
        meta.repo_id, len(rename_map), output_root,
    )
    return output_root


# =====================================================================================
# CLI
# =====================================================================================
def _add_merge_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "merge",
        help="Merge multiple compatible LeRobot v2.1 datasets into one.",
        description="Merge multiple compatible LeRobot v2.1 datasets into one.",
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Local root directories of the source datasets (>=2).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination directory for the merged dataset (must not exist or be empty).",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="repo_id recorded in the output info.json (default: output dir name).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="(TODO, not implemented in v1) Push the merged dataset to the Hugging Face Hub.",
    )
    p.set_defaults(func=_cmd_merge)


def _cmd_merge(args: argparse.Namespace) -> int:
    if args.push_to_hub:
        logger.warning("--push-to-hub is not implemented in v1; skipping push.")
    merge_datasets(inputs=args.inputs, output=args.output, repo_id=args.repo_id)
    return 0


def _add_delete_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "delete-episodes",
        help="Delete episodes from a LeRobot v2.1 dataset (survivors re-indexed 0..N-1).",
        description=(
            "Delete episodes from a LeRobot v2.1 dataset. Surviving episodes are "
            "re-indexed contiguously from 0; per-frame index/episode_index columns "
            "and meta files are rebuilt."
        ),
    )
    p.add_argument(
        "--input",
        required=True,
        help="Source dataset root directory (contains meta/info.json).",
    )
    p.add_argument(
        "--episode-indices",
        nargs="+",
        required=True,
        type=int,
        help="Source episode_index values to delete (space-separated, e.g. --episode-indices 0 3 5).",
    )
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--output",
        default=None,
        help="Destination directory for the result (must not exist or be empty).",
    )
    out_group.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Write the result back into --input. The original is first moved aside to "
            "<input>_backup_<timestamp> for reversibility."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="repo_id recorded in the output info.json (default: keep source repo_id).",
    )
    p.set_defaults(func=_cmd_delete)


def _cmd_delete(args: argparse.Namespace) -> int:
    delete_episodes(
        input=args.input,
        episode_indices=args.episode_indices,
        output=args.output,
        repo_id=args.repo_id,
        in_place=args.in_place,
    )
    return 0


def _add_rename_cameras_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "rename-cameras",
        help="Rename camera feature keys in a LeRobot v2.1 dataset.",
        description=(
            "Rename camera (video/image) feature keys. Updates info features, "
            "stats, episodes_stats, video directory names, and image parquet "
            "columns when applicable. Short names without '.' are prefixed with "
            "observation.images."
        ),
    )
    p.add_argument(
        "--input",
        required=True,
        help="Source dataset root directory (contains meta/info.json).",
    )
    p.add_argument(
        "--rename",
        action="append",
        required=True,
        metavar="OLD=NEW",
        help=(
            "Camera rename mapping (repeatable). Accepts short names "
            "(e.g. left_cam=cam_left) or full feature keys "
            "(e.g. observation.images.left_cam=observation.images.cam_left)."
        ),
    )
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--output",
        default=None,
        help="Destination directory for the result (must not exist or be empty).",
    )
    out_group.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Write the result back into --input. The original is first moved aside to "
            "<input>_backup_<timestamp> for reversibility."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="repo_id recorded in the output info.json (default: keep source repo_id).",
    )
    p.set_defaults(func=_cmd_rename_cameras)


def _cmd_rename_cameras(args: argparse.Namespace) -> int:
    rename_cameras(
        input=args.input,
        rename=args.rename,
        output=args.output,
        repo_id=args.repo_id,
        in_place=args.in_place,
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-edit-dataset",
        description=(
            "Edit LeRobot datasets (format v2.1). "
            "Sub-commands: merge, delete-episodes, rename-cameras."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    _add_merge_parser(subparsers)
    _add_delete_parser(subparsers)
    _add_rename_cameras_parser(subparsers)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = make_parser()
    args = parser.parse_args()
    try:
        rc = args.func(args)
    except Exception as e:  # noqa: BLE001
        logger.error("%s: %s", type(e).__name__, e)
        sys.exit(1)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
