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

# 6 cặp khớp trái/phải
PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

JOINTS = [joint for pair in PAIRS for joint in pair]

SWAPPED_JOINTS = [
    joint
    for left, right in PAIRS
    for joint in (right, left)
]


def pose(row, joint_names):
    """Lấy tọa độ 3D của 12 khớp."""
    return np.array([
        [
            row[f"{joint}_x"],
            row[f"{joint}_y"],
            row[f"{joint}_z"],
        ]
        for joint in joint_names
    ], dtype=float)


def pa_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE, đơn vị mm."""
    valid = (
        np.isfinite(pred).all(axis=1)
        & np.isfinite(gt).all(axis=1)
    )

    X = pred[valid]
    Y = gt[valid]

    if len(X) < 8:
        return np.nan

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)

    Xc = X - mx
    Yc = Y - my

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

    aligned = scale * (Xc @ R) + my

    return np.linalg.norm(aligned - Y, axis=1).mean() * 1000


print("Loading GT...")
gt = pd.read_csv(GT_FILE)

# Chỉ giữ camera 0
gt = gt[gt["camera"] == 0].copy()
gt = gt.set_index("frame")

print("GT frames:", len(gt))

print("Loading WHAM...")
wham = pd.read_csv(WHAM_FILE)

print("WHAM detections:", len(wham))

# Chỉ giữ body_id 0 và 1
wham = wham[wham["body_id"].isin([0, 1])].copy()

# Tìm các frame có cả body_id 0 và body_id 1
frames_0 = set(
    wham.loc[wham["body_id"] == 0, "frame_id"]
)

frames_1 = set(
    wham.loc[wham["body_id"] == 1, "frame_id"]
)

common_frames = sorted(
    frames_0 & frames_1 & set(gt.index)
)

print("\nFrames containing both body_id 0 and 1:", len(common_frames))

results = []

for frame in common_frames:
    gt_row = gt.loc[frame]

    # GT CSV lưu tọa độ theo mm, chuyển sang mét
    gt_pose = pose(gt_row, JOINTS) / 1000.0

    frame_rows = wham[wham["frame_id"] == frame]

    for body_id in [0, 1]:
        candidates = frame_rows[
            frame_rows["body_id"] == body_id
        ]

        if len(candidates) != 1:
            continue

        wham_row = candidates.iloc[0]

        pred_normal = pose(wham_row, JOINTS)
        pred_swapped = pose(wham_row, SWAPPED_JOINTS)

        error_normal = pa_mpjpe(pred_normal, gt_pose)
        error_swapped = pa_mpjpe(pred_swapped, gt_pose)

        results.append({
            "frame": frame,
            "body_id": body_id,
            "pa_mpjpe_normal_mm": error_normal,
            "pa_mpjpe_swapped_mm": error_swapped,
        })

result_df = pd.DataFrame(results)

output_file = (
    "outputs/subject_matching/"
    "compare_bodyid_0_1_same_frames.csv"
)

result_df.to_csv(output_file, index=False)

print("\nSaved:", output_file)

print("\nMedian PA-MPJPE:")
print(
    result_df.groupby("body_id")[
        ["pa_mpjpe_normal_mm", "pa_mpjpe_swapped_mm"]
    ].median().to_string()
)

print("\nMean PA-MPJPE:")
print(
    result_df.groupby("body_id")[
        ["pa_mpjpe_normal_mm", "pa_mpjpe_swapped_mm"]
    ].mean().to_string()
)

print("\nNumber of frames evaluated:")
print(result_df.groupby("body_id").size())

print("\nHead-to-head comparison (normal mapping):")

pivot = result_df.pivot(
    index="frame",
    columns="body_id",
    values="pa_mpjpe_normal_mm"
).dropna()

print("Common evaluated frames:", len(pivot))
print("ID 0 wins:", int((pivot[0] < pivot[1]).sum()))
print("ID 1 wins:", int((pivot[1] < pivot[0]).sum()))
print("Ties:", int((pivot[0] == pivot[1]).sum()))

print("\nHead-to-head comparison (swapped mapping):")

pivot_swapped = result_df.pivot(
    index="frame",
    columns="body_id",
    values="pa_mpjpe_swapped_mm"
).dropna()

print("ID 0 wins:", int(
    (pivot_swapped[0] < pivot_swapped[1]).sum()
))
print("ID 1 wins:", int(
    (pivot_swapped[1] < pivot_swapped[0]).sum()
))

print("\nFirst 20 results:")
print(result_df.head(20).to_string(index=False))
