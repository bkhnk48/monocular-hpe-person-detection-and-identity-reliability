from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _build_pose_array(
    df: pd.DataFrame,
    joints: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert joint coordinate columns into an array of shape (N, J, 3).

    Missing or non-numeric coordinates are represented by NaN.
    """
    n_rows = len(df)
    n_joints = len(joints)

    poses = np.full((n_rows, n_joints, 3), np.nan, dtype=np.float64)

    for j_idx, joint in enumerate(joints):
        for axis_idx, axis in enumerate(("x", "y", "z")):
            column = f"{joint}_{axis}"

            if column not in df.columns:
                continue

            poses[:, j_idx, axis_idx] = pd.to_numeric(
                df[column], errors="coerce"
            ).to_numpy(dtype=np.float64)

    valid_joints = np.isfinite(poses).all(axis=2)
    valid_joint_count = valid_joints.sum(axis=1)

    return poses, valid_joint_count


def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    source_name: str,
) -> None:
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def load_ground_truth(
    path: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load GT poses.

    Output includes:
      - frame
      - person_id
      - pose: ndarray with shape (J, 3), in meters
      - valid_joint_count
      - source_row
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"GT file not found: {path}")

    df = pd.read_csv(path)

    gt_cfg = config["gt"]
    frame_col = gt_cfg["frame_column"]
    camera_col = gt_cfg.get("camera_column")
    camera_value = gt_cfg.get("camera_value")

    _validate_columns(df, [frame_col], "GT")

    # Filter camera if the configured camera column exists.
    if camera_col and camera_value is not None:
        _validate_columns(df, [camera_col], "GT")
        df = df[df[camera_col] == camera_value].copy()

    df.reset_index(drop=True, inplace=True)
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    df["frame"] = pd.to_numeric(df[frame_col], errors="raise").astype(int)

    person_col = gt_cfg.get("person_id_column")

    if person_col and person_col in df.columns:
        df["person_id"] = df[person_col]
    elif gt_cfg.get("allow_single_person_fallback", False):
        fallback_id = gt_cfg.get("fallback_person_id", 0)
        df["person_id"] = fallback_id
    else:
        raise ValueError(
            "GT has no person identity column and fallback is disabled."
        )

    joints = config["joints"]["shared"]
    poses, valid_counts = _build_pose_array(df, joints)

    # Convert GT coordinates to meters.
    unit = gt_cfg.get("coordinate_unit", "m").lower()

    if unit == "mm":
        poses = poses / 1000.0
    elif unit != "m":
        raise ValueError(f"Unsupported GT coordinate unit: {unit}")

    df["pose"] = list(poses)
    df["valid_joint_count"] = valid_counts

    return df


def load_detections(
    path: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load WHAM detections.

    Each input row receives a unique detection_uid.
    body_id is preserved separately as the WHAM-provided ID.

    Output includes:
      - frame
      - body_id
      - detection_uid
      - pose: ndarray with shape (J, 3), in meters
      - valid_joint_count
      - source_row
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Detection file not found: {path}")

    df = pd.read_csv(path)

    det_cfg = config["detections"]
    frame_col = det_cfg["frame_column"]
    detection_id_col = det_cfg.get("detection_id_column")

    required = [frame_col]
    if detection_id_col:
        required.append(detection_id_col)

    _validate_columns(df, required, "Detections")

    df.reset_index(drop=True, inplace=True)
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    df["frame"] = pd.to_numeric(df[frame_col], errors="raise").astype(int)

    if detection_id_col:
        df["body_id"] = df[detection_id_col]
    else:
        df["body_id"] = np.nan

    # Unique row-level ID; do not assume body_id is globally unique.
    df["detection_uid"] = df["source_row"].astype(str)

    joints = config["joints"]["shared"]
    poses, valid_counts = _build_pose_array(df, joints)

    unit = det_cfg.get("coordinate_unit", "m").lower()

    if unit == "mm":
        poses = poses / 1000.0
    elif unit != "m":
        raise ValueError(f"Unsupported detection coordinate unit: {unit}")

    df["pose"] = list(poses)
    df["valid_joint_count"] = valid_counts

    return df


def load_dataset(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both GT and WHAM detections using the same configuration."""
    gt = load_ground_truth(config["data"]["gt_file"], config)
    detections = load_detections(config["data"]["detections_file"], config)

    return gt, detections
