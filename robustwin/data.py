from __future__ import annotations

import math
from pathlib import Path

from .common import EPISODES_PER_TASK, TASKS, canonical_hash, read_json, sha256

FEATURES = ("observation.state", "action")
CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")


def audit_data(root: Path, manifest: Path, *, deep: bool = False) -> dict:
    """Validate the AMD clean-data layout. Deep mode also scans all state/action rows."""
    entries = {}
    expected_root = (root / "data/lerobot").resolve()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] != "robotwin":
            raise ValueError(f"Manifest line {line_number}: expected 'robotwin /absolute/dataset/path'.")
        path = Path(parts[1])
        if not path.is_absolute():
            raise ValueError(f"Dataset path must be absolute: {path}")
        task = path.name.removesuffix("_joint_v30")
        if task not in TASKS or path.name != task + "_joint_v30":
            raise ValueError(f"Unexpected task or non-clean AMD dataset: {path}")
        if path.resolve() != (expected_root / f"{task}_joint_v30").resolve():
            raise ValueError(f"Dataset is outside the designated AMD clean-data location: {path}")
        if task in entries:
            raise ValueError(f"Duplicate task: {task}")
        if not (root / "data/demo_clean" / task).is_dir():
            raise ValueError(f"Missing clean source directory for {task}")
        info_path = path / "meta/info.json"
        info = read_json(info_path)
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"{task}: expected the AMD image's LeRobot v3.0 conversion.")
        if info.get("total_episodes") != EPISODES_PER_TASK:
            raise ValueError(f"{task}: expected exactly {EPISODES_PER_TASK} training episodes.")
        frames = info.get("total_frames")
        if type(frames) is not int or frames < EPISODES_PER_TASK:
            raise ValueError(f"{task}: invalid total_frames.")
        features = info.get("features", {})
        for feature in FEATURES:
            if features.get(feature, {}).get("shape") != [14]:
                raise ValueError(f"{task}: {feature} must have 14 dimensions.")
        for feature in CAMERAS:
            if features.get(feature, {}).get("dtype") != "video":
                raise ValueError(f"{task}: missing video feature {feature}.")
            if not any((path / "videos" / feature).rglob("*.mp4")):
                raise ValueError(f"{task}: missing video files for {feature}.")
        if not list((path / "data").rglob("*.parquet")):
            raise ValueError(f"{task}: no parquet data.")
        if deep:
            audit_rows(path, info)
        entries[task] = {"path": str(path.resolve()), "episodes": EPISODES_PER_TASK,
                         "frames": frames, "info_sha256": sha256(info_path)}
    if set(entries) != set(TASKS):
        raise ValueError(f"Missing tasks: {sorted(set(TASKS) - set(entries))}")
    ordered = {task: entries[task] for task in TASKS}
    return {"schema_version": 1, "setting": "demo_clean", "tasks": ordered,
            "task_count": len(entries), "episode_count": len(entries) * EPISODES_PER_TASK,
            "scan": "all_state_action_rows" if deep else "metadata_and_file_presence",
            "fingerprint": canonical_hash(ordered),
            "provenance_limit": "Checks the supplied AMD clean-data layout; does not cryptographically certify the dataset's origin or decode every video frame."}


def audit_rows(path: Path, info: dict) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Deep audit requires pyarrow from /opt/robotwin-env; run it inside the AMD image.") from exc
    counts: dict[int, int] = {}
    seen: dict[int, set[int]] = {}
    frame_count = 0
    for file in sorted((path / "data").rglob("*.parquet")):
        reader = pq.ParquetFile(file)
        for batch in reader.iter_batches(batch_size=2048, columns=["episode_index", "frame_index", *FEATURES]):
            for row in batch.to_pylist():
                episode, frame = row["episode_index"], row["frame_index"]
                if type(episode) is not int or not 0 <= episode < EPISODES_PER_TASK or type(frame) is not int or frame < 0:
                    raise ValueError(f"Invalid episode/frame index in {file}")
                episode_frames = seen.setdefault(episode, set())
                if frame in episode_frames:
                    raise ValueError(f"Duplicate episode/frame {episode}/{frame} in {file}")
                episode_frames.add(frame)
                counts[episode] = counts.get(episode, 0) + 1
                for key in FEATURES:
                    values = row[key]
                    if not isinstance(values, list) or len(values) != 14 or not all(
                        isinstance(value, (int, float)) and math.isfinite(value) for value in values
                    ):
                        raise ValueError(f"Non-finite or invalid {key} in {file}, episode {episode}, frame {frame}")
                frame_count += 1
    if frame_count != info["total_frames"] or set(counts) != set(range(EPISODES_PER_TASK)):
        raise ValueError(f"Frame/episode totals do not match metadata: {path}")
    if any(frames != set(range(counts[episode])) for episode, frames in seen.items()):
        raise ValueError(f"Non-contiguous frame indices: {path}")
