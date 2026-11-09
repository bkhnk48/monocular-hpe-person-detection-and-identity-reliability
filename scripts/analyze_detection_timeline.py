from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCORES_PATH = Path(
    "outputs/subject_matching/all_bodyid_frame_scores.csv"
)

GT_PATH = Path(
    "data/raw/mpi_inf_3dhp/ground_truth/S8_Seq1_video_cam0.csv"
)

OUTPUT_DIR = Path("outputs/subject_matching")

BEST_FRAME_PATH = OUTPUT_DIR / "best_detection_per_frame.csv"
SEGMENTS_PATH = OUTPUT_DIR / "bodyid_temporal_segments.csv"


# ============================================================
# LOAD DATA
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANALYZE WHAM DETECTION TIMELINE")
    print("=" * 70)

    print("\nLoading GT...")
    gt = pd.read_csv(GT_PATH)

    # Chỉ lấy camera 0
    gt = gt[gt["camera"] == 0].copy()

    gt["frame"] = pd.to_numeric(gt["frame"], errors="coerce")
    gt = gt.dropna(subset=["frame"])

    gt_frames = sorted(gt["frame"].astype(int).unique())

    print("GT camera 0 frames:", len(gt_frames))
    print("GT frame range:", min(gt_frames), "-", max(gt_frames))

    print("\nLoading PA-MPJPE scores...")
    scores = pd.read_csv(SCORES_PATH)

    required_columns = ["frame_id", "body_id", "pa_mpjpe_mm"]

    missing_columns = [
        col for col in required_columns
        if col not in scores.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in scores CSV: {missing_columns}"
        )

    scores["frame_id"] = pd.to_numeric(
        scores["frame_id"], errors="coerce"
    )

    scores["body_id"] = pd.to_numeric(
        scores["body_id"], errors="coerce"
    )

    scores["pa_mpjpe_mm"] = pd.to_numeric(
        scores["pa_mpjpe_mm"], errors="coerce"
    )

    scores = scores.dropna(
        subset=["frame_id", "body_id", "pa_mpjpe_mm"]
    ).copy()

    scores["frame_id"] = scores["frame_id"].astype(int)
    scores["body_id"] = scores["body_id"].astype(int)

    print("Scored detections:", len(scores))
    print("Unique WHAM frames:", scores["frame_id"].nunique())
    print("Unique body IDs:", sorted(scores["body_id"].unique()))

    # Kiểm tra mỗi body_id chỉ có tối đa một detection trong một frame
    duplicate_count = scores.duplicated(
        subset=["frame_id", "body_id"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate frame_id/body_id rows. "
            "Please inspect the input CSV before continuing."
        )

    # Chỉ giữ các frame thuộc GT camera 0
    scores = scores[
        scores["frame_id"].isin(gt_frames)
    ].copy()


    # ========================================================
    # PART 1: BEST DETECTION PER FRAME
    # ========================================================

    print("\n" + "=" * 70)
    print("PART 1: BEST DETECTION PER FRAME")
    print("=" * 70)

    # Đếm số detection trong từng frame
    detection_counts = (
        scores.groupby("frame_id")
        .size()
        .rename("detection_count")
    )

    # Sắp xếp PA-MPJPE tăng dần trong từng frame
    ranked = scores.sort_values(
        ["frame_id", "pa_mpjpe_mm", "body_id"]
    ).copy()

    ranked["rank_in_frame"] = (
        ranked.groupby("frame_id").cumcount() + 1
    )

    # Detection tốt nhất
    best = ranked[ranked["rank_in_frame"] == 1][
        ["frame_id", "body_id", "pa_mpjpe_mm"]
    ].rename(
        columns={
            "body_id": "best_body_id",
            "pa_mpjpe_mm": "best_pa_mpjpe_mm",
        }
    )

    # Detection tốt thứ hai, nếu có
    second = ranked[ranked["rank_in_frame"] == 2][
        ["frame_id", "body_id", "pa_mpjpe_mm"]
    ].rename(
        columns={
            "body_id": "second_best_body_id",
            "pa_mpjpe_mm": "second_best_pa_mpjpe_mm",
        }
    )

    # Khởi tạo bảng đầy đủ tất cả frame GT
    best_per_frame = pd.DataFrame({
        "frame_id": gt_frames
    })

    best_per_frame = best_per_frame.merge(
        detection_counts,
        on="frame_id",
        how="left"
    )

    best_per_frame = best_per_frame.merge(
        best,
        on="frame_id",
        how="left"
    )

    best_per_frame = best_per_frame.merge(
        second,
        on="frame_id",
        how="left"
    )

    best_per_frame["detection_count"] = (
        best_per_frame["detection_count"]
        .fillna(0)
        .astype(int)
    )

    best_per_frame["has_detection"] = (
        best_per_frame["detection_count"] > 0
    )

    # Khoảng cách giữa score tốt nhất và tốt thứ hai
    best_per_frame["second_minus_best_mm"] = (
        best_per_frame["second_best_pa_mpjpe_mm"]
        - best_per_frame["best_pa_mpjpe_mm"]
    )

    best_per_frame.to_csv(
        BEST_FRAME_PATH,
        index=False
    )

    print("Saved:", BEST_FRAME_PATH)
    print("Total GT frames:", len(best_per_frame))

    frames_with_detection = (
        best_per_frame["has_detection"].sum()
    )

    frames_without_detection = (
        len(best_per_frame) - frames_with_detection
    )

    print("Frames with detection:", frames_with_detection)
    print("Frames without detection:", frames_without_detection)

    print("\nDetection count distribution:")
    print(
        best_per_frame["detection_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBest body_id distribution:")
    print(
        best_per_frame["best_body_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBest body_id summary:")
    best_summary = (
        best_per_frame.dropna(subset=["best_body_id"])
        .groupby("best_body_id")
        .agg(
            best_frames=("frame_id", "count"),
            median_best_pa_mpjpe_mm=(
                "best_pa_mpjpe_mm", "median"
            ),
            mean_best_pa_mpjpe_mm=(
                "best_pa_mpjpe_mm", "mean"
            ),
        )
        .reset_index()
    )

    best_summary["best_body_id"] = (
        best_summary["best_body_id"].astype(int)
    )

    print(best_summary.to_string(index=False))


    # ========================================================
    # PART 2: TEMPORAL SEGMENTS PER BODY ID
    # ========================================================

    print("\n" + "=" * 70)
    print("PART 2: TEMPORAL SEGMENTS PER BODY ID")
    print("=" * 70)

    segment_records = []

    for body_id, group in scores.groupby("body_id"):

        group = group.sort_values("frame_id").copy()

        # Tách đoạn mới nếu khoảng cách frame lớn hơn 1
        frame_differences = group["frame_id"].diff()

        group["segment_id"] = (
            frame_differences.gt(1).cumsum()
        )

        for segment_id, segment in group.groupby("segment_id"):

            frames = segment["frame_id"].astype(int)

            start_frame = int(frames.min())
            end_frame = int(frames.max())

            record = {
                "body_id": int(body_id),
                "segment_id": int(segment_id) + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "number_of_frames": int(len(segment)),
                "frame_span": end_frame - start_frame + 1,
                "median_pa_mpjpe_mm": float(
                    segment["pa_mpjpe_mm"].median()
                ),
                "mean_pa_mpjpe_mm": float(
                    segment["pa_mpjpe_mm"].mean()
                ),
                "min_pa_mpjpe_mm": float(
                    segment["pa_mpjpe_mm"].min()
                ),
                "max_pa_mpjpe_mm": float(
                    segment["pa_mpjpe_mm"].max()
                ),
            }

            segment_records.append(record)

    segments = pd.DataFrame(segment_records)

    if not segments.empty:
        segments = segments.sort_values(
            ["body_id", "start_frame"]
        ).reset_index(drop=True)

    segments.to_csv(
        SEGMENTS_PATH,
        index=False
    )

    print("Saved:", SEGMENTS_PATH)
    print("Total temporal segments:", len(segments))

    print("\nNumber of segments per body_id:")

    segment_summary = (
        segments.groupby("body_id")
        .agg(
            number_of_segments=("segment_id", "count"),
            total_detection_frames=("number_of_frames", "sum"),
            longest_segment_frames=("number_of_frames", "max"),
            median_segment_length=("number_of_frames", "median"),
            median_pa_mpjpe_mm=("median_pa_mpjpe_mm", "median"),
        )
        .reset_index()
    )

    print(segment_summary.to_string(index=False))


    # ========================================================
    # PART 3: LONGEST SEGMENTS
    # ========================================================

    print("\n" + "=" * 70)
    print("PART 3: LONGEST TEMPORAL SEGMENTS")
    print("=" * 70)

    longest = segments.sort_values(
        "number_of_frames",
        ascending=False
    ).head(20)

    print(
        longest[
            [
                "body_id",
                "segment_id",
                "start_frame",
                "end_frame",
                "number_of_frames",
                "median_pa_mpjpe_mm",
            ]
        ].to_string(index=False)
    )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("Output files:")
    print("1.", BEST_FRAME_PATH)
    print("2.", SEGMENTS_PATH)

    print(
        "\nIMPORTANT: The lowest PA-MPJPE detection in a frame "
        "is only the best geometric match to GT among available "
        "detections. It is NOT automatically a verified true positive."
    )


if __name__ == "__main__":
    main()
