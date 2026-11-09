import itertools
import os
import numpy as np
import pandas as pd

GT_FILE = (
    "data/raw/mpi_inf_3dhp/ground_truth/"
    "S8_Seq1_video_cam0.csv"
)

WHAM_FILE = (
    "data/raw/mpi_inf_3dhp/wham_predictions/"
    "RA-WHAM_S8_Seq1_video_cam0.csv"
)

OUTPUT_DIR = "outputs/subject_matching"

PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

JOINTS = [joint for pair in PAIRS for joint in pair]


def get_pose(row, joint_names):
    return np.array([
        [
            row[f"{joint}_x"],
            row[f"{joint}_y"],
            row[f"{joint}_z"],
        ]
        for joint in joint_names
    ], dtype=float)


def pa_mpjpe(pred, gt):
    """PA-MPJPE, trả về đơn vị mm."""

    valid = (
        np.isfinite(pred).all(axis=1)
        & np.isfinite(gt).all(axis=1)
    )

    X = pred[valid]
    Y = gt[valid]

    if len(X) < 8:
        return np.nan

    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)

    Xc = X - X_mean
    Yc = Y - Y_mean

    U, S, Vt = np.linalg.svd(Xc.T @ Yc)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt

    denominator = np.sum(Xc ** 2)

    if denominator < 1e-12:
        return np.nan

    scale = S.sum() / denominator

    aligned = scale * (Xc @ R) + Y_mean

    return np.linalg.norm(aligned - Y, axis=1).mean() * 1000


# --------------------------------------------------
# 1. Load GT
# --------------------------------------------------

print("Loading GT...")

gt = pd.read_csv(GT_FILE)

gt = gt[gt["camera"] == 0].copy()
gt = gt.set_index("frame")

print("GT frames:", len(gt))


# --------------------------------------------------
# 2. Load WHAM
# --------------------------------------------------

print("\nLoading WHAM...")

wham = pd.read_csv(WHAM_FILE)

print("WHAM detections:", len(wham))
print("Body IDs:", sorted(wham["body_id"].unique()))


# Kiểm tra mỗi body_id có tối đa một detection trong một frame
duplicates = wham.duplicated(
    subset=["frame_id", "body_id"],
    keep=False
)

if duplicates.any():
    print("\nERROR: Duplicate frame_id/body_id combinations:")
    print(
        wham.loc[
            duplicates,
            ["frame_id", "body_id"]
        ].head(20).to_string(index=False)
    )
    raise ValueError(
        "Có detection trùng frame_id và body_id. "
        "Cần xử lý trước khi so sánh."
    )


# --------------------------------------------------
# 3. Compute PA-MPJPE for every detection
# --------------------------------------------------

print("\nComputing PA-MPJPE for all detections...")

records = []

for frame_id, frame_rows in wham.groupby("frame_id"):

    if frame_id not in gt.index:
        continue

    gt_row = gt.loc[frame_id]

    # GT tọa độ mm -> m
    gt_pose = get_pose(gt_row, JOINTS) / 1000.0

    for _, row in frame_rows.iterrows():

        pred_pose = get_pose(row, JOINTS)

        error = pa_mpjpe(pred_pose, gt_pose)

        records.append({
            "frame_id": int(frame_id),
            "body_id": int(row["body_id"]),
            "pa_mpjpe_mm": error,
        })


scores = pd.DataFrame(records)

os.makedirs(OUTPUT_DIR, exist_ok=True)

scores_file = os.path.join(
    OUTPUT_DIR,
    "all_bodyid_frame_scores.csv"
)

scores.to_csv(scores_file, index=False)

print("\nSaved:", scores_file)
print("Scored detections:", len(scores))
print("Scored frames:", scores["frame_id"].nunique())


# --------------------------------------------------
# 4. Summary by body_id
# --------------------------------------------------

print("\n===== SUMMARY BY BODY_ID =====")

summary = (
    scores.groupby("body_id")
    .agg(
        detection_frames=("frame_id", "nunique"),
        median_pa_mpjpe_mm=("pa_mpjpe_mm", "median"),
        mean_pa_mpjpe_mm=("pa_mpjpe_mm", "mean"),
        std_pa_mpjpe_mm=("pa_mpjpe_mm", "std"),
    )
    .sort_values("median_pa_mpjpe_mm")
)

print(summary.to_string())


# --------------------------------------------------
# 5. Pairwise comparison on identical frames
# --------------------------------------------------

print("\n===== PAIRWISE COMPARISON =====")

pair_records = []

# Mỗi frame có một dictionary:
# body_id -> PA-MPJPE
for frame_id, group in scores.groupby("frame_id"):

    errors = dict(zip(
        group["body_id"],
        group["pa_mpjpe_mm"]
    ))

    # Chỉ giữ các ID có sai số hợp lệ
    errors = {
        body_id: error
        for body_id, error in errors.items()
        if np.isfinite(error)
    }

    if len(errors) < 2:
        continue

    for id_a, id_b in itertools.combinations(
        sorted(errors.keys()), 2
    ):

        error_a = errors[id_a]
        error_b = errors[id_b]

        pair_records.append({
            "frame_id": int(frame_id),
            "body_id_a": int(id_a),
            "body_id_b": int(id_b),
            "error_a_mm": error_a,
            "error_b_mm": error_b,
            "difference_b_minus_a_mm": error_b - error_a,
        })


pair_frames = pd.DataFrame(pair_records)

if pair_frames.empty:
    raise ValueError("Không tìm thấy cặp body_id cùng xuất hiện.")


# Tổng hợp từng cặp
pair_summary = []

for (id_a, id_b), group in pair_frames.groupby(
    ["body_id_a", "body_id_b"]
):

    wins_a = int(
        (group["error_a_mm"] < group["error_b_mm"]).sum()
    )

    wins_b = int(
        (group["error_b_mm"] < group["error_a_mm"]).sum()
    )

    ties = int(
        (group["error_a_mm"] == group["error_b_mm"]).sum()
    )

    pair_summary.append({
        "body_id_a": id_a,
        "body_id_b": id_b,
        "common_frames": len(group),
        "id_a_wins": wins_a,
        "id_b_wins": wins_b,
        "ties": ties,
        "id_a_win_rate": wins_a / len(group),
        "id_b_win_rate": wins_b / len(group),
        "median_error_a_mm": group["error_a_mm"].median(),
        "median_error_b_mm": group["error_b_mm"].median(),
        "median_difference_b_minus_a_mm":
            group["difference_b_minus_a_mm"].median(),
    })


pair_summary = pd.DataFrame(pair_summary)

pair_summary = pair_summary.sort_values(
    ["common_frames", "id_a_win_rate"],
    ascending=[False, False]
)

pair_file = os.path.join(
    OUTPUT_DIR,
    "pairwise_bodyid_comparison.csv"
)

pair_summary.to_csv(pair_file, index=False)

print("\nSaved:", pair_file)

print("\nPairwise results:")
print(pair_summary.to_string(index=False))


print("\n===== DONE =====")
