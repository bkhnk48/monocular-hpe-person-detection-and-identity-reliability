
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GT_PATH = (
    ROOT
    / "data/processed/mpi_inf_3dhp/gt_frame_status.csv"
)

WHAM_PATH = (
    ROOT
    / "data/raw/mpi_inf_3dhp/wham_predictions/"
    / "RA-WHAM_S8_Seq1_video_cam0.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs/gt_wham_frame_coverage.csv"
)

gt = pd.read_csv(GT_PATH)
wham = pd.read_csv(WHAM_PATH)

# Count WHAM detections in each frame.
wham_by_frame = (
    wham.groupby("frame_id")
    .agg(
        wham_detection_count=("body_id", "size"),
        wham_body_ids=(
            "body_id",
            lambda x: ",".join(
                map(str, sorted(x.unique()))
            ),
        ),
    )
    .reset_index()
)

# Merge GT frame information with WHAM detections.
df = gt.merge(
    wham_by_frame,
    on="frame_id",
    how="left",
)

df["wham_detection_count"] = (
    df["wham_detection_count"]
    .fillna(0)
    .astype(int)
)

df["wham_body_ids"] = (
    df["wham_body_ids"]
    .fillna("")
)

df["wham_multiple_detections"] = (
    df["wham_detection_count"] > 1
)

# Describe GT image-space coverage.
df["gt_image_status"] = "all_joint_projections_in_image"

df.loc[
    df["visible_joint_count"] == 0,
    "gt_image_status"
] = "no_joint_in_image"

df.loc[
    (df["visible_joint_count"] > 0)
    & (df["visible_joint_count"] < 28),
    "gt_image_status"
] = "partial_joint_projection"

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Total frames:", len(df))

print("\n=== WHAM DETECTION COUNT ===")
print(
    df["wham_detection_count"]
    .value_counts()
    .sort_index()
)

print("\n=== DETECTIONS BY GT IMAGE STATUS ===")

summary = (
    df.groupby("gt_image_status")
    .agg(
        frames=("frame_id", "count"),
        no_detection=(
            "wham_detection_count",
            lambda x: (x == 0).sum(),
        ),
        one_detection=(
            "wham_detection_count",
            lambda x: (x == 1).sum(),
        ),
        multiple_detections=(
            "wham_detection_count",
            lambda x: (x > 1).sum(),
        ),
    )
)

print(summary)

print("\n=== FRAMES WITH MULTIPLE WHAM DETECTIONS ===")
print(
    df.loc[
        df["wham_multiple_detections"],
        [
            "frame_id",
            "visible_joint_count",
            "wham_detection_count",
            "wham_body_ids",
        ],
    ].head(30).to_string(index=False)
)
