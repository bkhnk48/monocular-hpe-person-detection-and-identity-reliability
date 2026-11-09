from typing import Any

import numpy as np
import pandas as pd


def validate_dataset(
    gt: pd.DataFrame,
    detections: pd.DataFrame,
    config: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate GT and detection tables before association.

    Errors indicate structural problems that can invalidate evaluation.
    Warnings indicate unusual conditions that should be inspected.
    """
    errors = []
    warnings = []

    validation_cfg = config["validation"]
    min_joints = validation_cfg.get("min_valid_joints_per_pose", 1)

    # ---------------------------------------------------------
    # 1. Required columns
    # ---------------------------------------------------------
    gt_required = {
        "frame",
        "person_id",
        "pose",
        "valid_joint_count",
    }
    det_required = {
        "frame",
        "body_id",
        "detection_uid",
        "pose",
        "valid_joint_count",
    }

    missing_gt = gt_required - set(gt.columns)
    missing_det = det_required - set(detections.columns)

    if missing_gt:
        errors.append(f"GT missing columns: {sorted(missing_gt)}")

    if missing_det:
        errors.append(
            f"Detections missing columns: {sorted(missing_det)}"
        )

    if errors:
        report = {
            "passed": False,
            "errors": errors,
            "warnings": warnings,
        }
        if strict:
            raise ValueError("\n".join(errors))
        return report

    # ---------------------------------------------------------
    # 2. Missing frame / identity
    # ---------------------------------------------------------
    if gt["frame"].isna().any():
        errors.append("GT contains missing frame values.")

    if gt["person_id"].isna().any():
        errors.append("GT contains missing person_id values.")

    if detections["frame"].isna().any():
        errors.append("Detections contain missing frame values.")

    if detections["detection_uid"].isna().any():
        errors.append("Detections contain missing detection_uid values.")

    # ---------------------------------------------------------
    # 3. Duplicate GT identity in the same frame
    # ---------------------------------------------------------
    if validation_cfg.get("check_duplicate_gt_identity_per_frame", True):
        duplicate_gt = gt.duplicated(
            subset=["frame", "person_id"],
            keep=False,
        )

        duplicate_gt_count = int(duplicate_gt.sum())

        if duplicate_gt_count:
            examples = (
                gt.loc[duplicate_gt, ["frame", "person_id"]]
                .head(10)
                .to_dict("records")
            )
            errors.append(
                "Duplicate GT (frame, person_id) pairs: "
                f"{duplicate_gt_count} rows. Examples: {examples}"
            )
    else:
        duplicate_gt_count = 0

    # ---------------------------------------------------------
    # 4. Duplicate detection identity in the same frame
    # ---------------------------------------------------------
    if validation_cfg.get("check_duplicate_detection_per_frame", True):
        duplicate_det = detections.duplicated(
            subset=["frame", "body_id"],
            keep=False,
        )

        # Ignore rows whose body_id is missing.
        duplicate_det &= detections["body_id"].notna()

        duplicate_det_count = int(duplicate_det.sum())

        if duplicate_det_count:
            examples = (
                detections.loc[duplicate_det, ["frame", "body_id"]]
                .head(10)
                .to_dict("records")
            )
            errors.append(
                "Duplicate detection (frame, body_id) pairs: "
                f"{duplicate_det_count} rows. Examples: {examples}"
            )
    else:
        duplicate_det_count = 0

    # detection_uid must identify exactly one input row.
    duplicate_uid_count = int(
        detections["detection_uid"].duplicated(keep=False).sum()
    )

    if duplicate_uid_count:
        errors.append(
            "Duplicate detection_uid values: "
            f"{duplicate_uid_count} rows."
        )

    # ---------------------------------------------------------
    # 5. Valid joint count
    # ---------------------------------------------------------
    invalid_gt_joints = (
        gt["valid_joint_count"].fillna(0) < min_joints
    )
    invalid_det_joints = (
        detections["valid_joint_count"].fillna(0) < min_joints
    )

    invalid_gt_count = int(invalid_gt_joints.sum())
    invalid_det_count = int(invalid_det_joints.sum())

    if invalid_gt_count:
        warnings.append(
            f"{invalid_gt_count} GT poses have fewer than "
            f"{min_joints} valid joints."
        )

    if invalid_det_count:
        warnings.append(
            f"{invalid_det_count} detections have fewer than "
            f"{min_joints} valid joints."
        )

    # ---------------------------------------------------------
    # 6. Pose shape and finite coordinates
    # ---------------------------------------------------------
    def inspect_poses(
        df: pd.DataFrame,
        source_name: str,
    ) -> tuple[int, int]:
        invalid_shape_count = 0
        nonfinite_pose_count = 0

        for index, pose in enumerate(df["pose"]):
            arr = np.asarray(pose)

            if arr.ndim != 2 or arr.shape != (12, 3):
                invalid_shape_count += 1
                continue

            if not np.isfinite(arr).all():
                nonfinite_pose_count += 1

        if invalid_shape_count:
            errors.append(
                f"{source_name}: {invalid_shape_count} poses "
                "do not have shape (12, 3)."
            )

        if nonfinite_pose_count:
            warnings.append(
                f"{source_name}: {nonfinite_pose_count} poses "
                "contain NaN or infinite coordinates."
            )

        return invalid_shape_count, nonfinite_pose_count

    invalid_gt_shape, nonfinite_gt = inspect_poses(gt, "GT")
    invalid_det_shape, nonfinite_det = inspect_poses(
        detections, "Detections"
    )

    # ---------------------------------------------------------
    # 7. Frame coverage
    # ---------------------------------------------------------
    gt_frames = set(gt["frame"].dropna().astype(int).unique())
    det_frames = set(detections["frame"].dropna().astype(int).unique())

    gt_without_detections = sorted(gt_frames - det_frames)
    detections_without_gt = sorted(det_frames - gt_frames)

    if gt_without_detections:
        warnings.append(
            f"{len(gt_without_detections)} GT frames have no detections."
        )

    if detections_without_gt:
        warnings.append(
            f"{len(detections_without_gt)} detection frames "
            "have no GT."
        )

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------
    report = {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "gt": {
            "rows": int(len(gt)),
            "unique_frames": int(gt["frame"].nunique()),
            "unique_person_ids": int(gt["person_id"].nunique()),
            "duplicate_identity_rows": duplicate_gt_count,
            "invalid_joint_count_rows": invalid_gt_count,
            "invalid_pose_shapes": invalid_gt_shape,
            "nonfinite_poses": nonfinite_gt,
        },
        "detections": {
            "rows": int(len(detections)),
            "unique_frames": int(detections["frame"].nunique()),
            "unique_body_ids": int(detections["body_id"].nunique()),
            "unique_detection_uids": int(
                detections["detection_uid"].nunique()
            ),
            "duplicate_identity_rows": duplicate_det_count,
            "duplicate_uid_rows": duplicate_uid_count,
            "invalid_joint_count_rows": invalid_det_count,
            "invalid_pose_shapes": invalid_det_shape,
            "nonfinite_poses": nonfinite_det,
        },
        "coverage": {
            "gt_frames_without_detections": len(gt_without_detections),
            "detection_frames_without_gt": len(detections_without_gt),
        },
    }

    if strict and errors:
        details = "\n".join(f"- {e}" for e in errors)
        raise ValueError(f"Dataset validation failed:\n{details}")

    return report


def print_validation_report(report: dict[str, Any]) -> None:
    """Print a readable validation report."""
    print("=" * 65)
    print("DATASET VALIDATION REPORT")
    print("=" * 65)

    print("Status:", "PASSED" if report["passed"] else "FAILED")

    for section in ("gt", "detections", "coverage"):
        if section in report:
            print(f"\n[{section.upper()}]")
            for key, value in report[section].items():
                print(f"{key}: {value}")

    if report["warnings"]:
        print("\n[WARNINGS]")
        for warning in report["warnings"]:
            print("-", warning)

    if report["errors"]:
        print("\n[ERRORS]")
        for error in report["errors"]:
            print("-", error)

    print("\n" + "=" * 65)
