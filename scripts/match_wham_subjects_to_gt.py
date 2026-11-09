import numpy as np
import pandas as pd
from pathlib import Path

GT_FILE = Path(
    "data/raw/mpi_inf_3dhp/ground_truth/S8_Seq1_video_cam0.csv"
)

WHAM_FILE = Path(
    "data/raw/mpi_inf_3dhp/wham_predictions/RA-WHAM_S8_Seq1_video_cam0.csv"
)

OUT_DIR = Path("outputs/subject_matching")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Các khớp có tên tương ứng trong cả GT và WHAM.
JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_pose(row, joints):
    """Lấy tọa độ 3D của các khớp từ một dòng CSV."""
    return np.array(
        [
            [row[f"{joint}_x"], row[f"{joint}_y"], row[f"{joint}_z"]]
            for joint in joints
        ],
        dtype=float,
    )


def pa_mpjpe(pred, gt):
    """
    Procrustes-aligned MPJPE.
    Căn chỉnh translation, rotation và scale trước khi tính sai số.
    Đơn vị đầu ra giống đơn vị của GT.
    """
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)

    X = pred[valid]
    Y = gt[valid]

    if len(X) < 8:
        return np.nan

    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)

    Xc = X - mean_X
    Yc = Y - mean_Y

    norm_X = np.sum(Xc ** 2)

    if norm_X < 1e-12:
        return np.nan

    U, S, Vt = np.linalg.svd(Xc.T @ Yc)

    R = U @ Vt

    # Đảm bảo phép quay không tạo phản chiếu.
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt

    scale = np.sum(S) / norm_X

    aligned = scale * (Xc @ R) + mean_Y

    errors = np.linalg.norm(aligned - Y, axis=1)

    return float(np.mean(errors))


def main():
    print("Loading GT:", GT_FILE)
    print("Loading WHAM:", WHAM_FILE)

    if not GT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy GT file: {GT_FILE}")

    if not WHAM_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy WHAM file: {WHAM_FILE}")

    gt = pd.read_csv(GT_FILE)
    wham = pd.read_csv(WHAM_FILE)

    required_gt = {"frame"}
    required_wham = {"frame_id", "body_id"}

    for joint in JOINTS:
        for axis in ["x", "y", "z"]:
            required_gt.add(f"{joint}_{axis}")
            required_wham.add(f"{joint}_{axis}")

    missing_gt = required_gt - set(gt.columns)
    missing_wham = required_wham - set(wham.columns)

    if missing_gt:
        raise ValueError(f"GT thiếu các cột: {sorted(missing_gt)}")

    if missing_wham:
        raise ValueError(f"WHAM thiếu các cột: {sorted(missing_wham)}")

    # GT CSV dùng mm; WHAM CSV dùng m.
    # Chuyển GT sang mét để thống nhất đơn vị.
    gt = gt.copy()
    wham = wham.copy()

    gt["frame"] = pd.to_numeric(gt["frame"], errors="coerce")
    wham["frame_id"] = pd.to_numeric(
        wham["frame_id"], errors="coerce"
    )
    wham["body_id"] = pd.to_numeric(
        wham["body_id"], errors="coerce"
    )

    for joint in JOINTS:
        for axis in ["x", "y", "z"]:
            gt[f"{joint}_{axis}"] = (
                pd.to_numeric(gt[f"{joint}_{axis}"], errors="coerce") / 1000.0
            )
            wham[f"{joint}_{axis}"] = pd.to_numeric(
                wham[f"{joint}_{axis}"], errors="coerce"
            )

    gt = gt.dropna(subset=["frame"])
    wham = wham.dropna(subset=["frame_id", "body_id"])

    gt["frame"] = gt["frame"].astype(int)
    wham["frame_id"] = wham["frame_id"].astype(int)
    wham["body_id"] = wham["body_id"].astype(int)

    gt_lookup = {
        int(row["frame"]): get_pose(row, JOINTS)
        for _, row in gt.iterrows()
    }

    records = []

    total = len(wham)

    for i, (_, row) in enumerate(wham.iterrows(), start=1):
        frame = int(row["frame_id"])
        body_id = int(row["body_id"])

        if frame not in gt_lookup:
            continue

        pred_pose = get_pose(row, JOINTS)
        gt_pose = gt_lookup[frame]

        error = pa_mpjpe(pred_pose, gt_pose)

        records.append({
            "frame": frame,
            "body_id": body_id,
            "pa_mpjpe_m": error,
            "pa_mpjpe_mm": error * 1000 if np.isfinite(error) else np.nan,
        })

        if i % 1000 == 0:
            print(f"Processed {i}/{total} WHAM detections")

    scores = pd.DataFrame(records)

    if scores.empty:
        raise RuntimeError(
            "Không có frame nào ghép được giữa WHAM và GT."
        )

    scores.to_csv(
        OUT_DIR / "frame_bodyid_scores.csv",
        index=False
    )

    # Tổng hợp sai số theo từng body_id.
    summary = (
        scores.groupby("body_id")
        .agg(
            detection_frames=("frame", "nunique"),
            median_pa_mpjpe_mm=("pa_mpjpe_mm", "median"),
            mean_pa_mpjpe_mm=("pa_mpjpe_mm", "mean"),
            std_pa_mpjpe_mm=("pa_mpjpe_mm", "std"),
        )
        .reset_index()
        .sort_values("median_pa_mpjpe_mm")
    )

    summary.to_csv(
        OUT_DIR / "subject_id_metrics.csv",
        index=False
    )

    # Tìm body_id có sai số thấp nhất trong từng frame.
    best_per_frame = (
        scores.sort_values("pa_mpjpe_mm")
        .groupby("frame", as_index=False)
        .first()
    )

    best_per_frame.to_csv(
        OUT_DIR / "subject_id_matching.csv",
        index=False
    )

    # Thống kê số frame mà mỗi body_id được chọn là tốt nhất.
    best_counts = (
        best_per_frame.groupby("body_id")
        .agg(
            best_frame_count=("frame", "nunique"),
            median_best_error_mm=("pa_mpjpe_mm", "median"),
        )
        .reset_index()
        .sort_values("best_frame_count", ascending=False)
    )

    best_counts.to_csv(
        OUT_DIR / "subject_id_best_counts.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("SUBJECT ID MATCHING RESULTS")
    print("=" * 65)

    print("\n1. PA-MPJPE theo body_id:")
    print(summary.to_string(index=False))

    print("\n2. Số frame mỗi body_id được chọn là tốt nhất:")
    print(best_counts.to_string(index=False))

    best_id = summary.iloc[0]["body_id"]

    print("\n3. Body ID có median PA-MPJPE thấp nhất:")
    print(f"body_id = {best_id}")
    print(
        "median PA-MPJPE = "
        f"{summary.iloc[0]['median_pa_mpjpe_mm']:.2f} mm"
    )

    print("\n4. Output files:")
    for filename in [
        "frame_bodyid_scores.csv",
        "subject_id_metrics.csv",
        "subject_id_matching.csv",
        "subject_id_best_counts.csv",
    ]:
        print(OUT_DIR / filename)

    print("\nLưu ý:")
    print(
        "PA-MPJPE đã căn chỉnh translation, rotation và scale "
        "riêng cho từng frame."
    )
    print(
        "Đây là phép so sánh hình dạng pose để hỗ trợ ghép subject ID, "
        "chưa phải MPJPE camera-space chuẩn."
    )


if __name__ == "__main__":
    main()
