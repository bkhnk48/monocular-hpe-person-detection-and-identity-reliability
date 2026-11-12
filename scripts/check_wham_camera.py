#!/usr/bin/env python3
"""Kiểm tra hệ tọa độ camera WHAM so với GT MPI hoặc GT video overlay.

Chế độ MPI đơn (tương thích cách chạy cũ):
  python3 scripts/check_wham_camera.py \\
      --pkl path/to/wham.pkl \\
      --annot data/raw/mpi_inf_3dhp/S8/annot.mat --camera 0 \\
      --mapping outputs/mapping/mpi_S8_Seq1_cam0/mapping_frames.csv

Chế độ video overlay:
  python3 scripts/check_wham_camera.py \\
      --pkl path/to/overlay_wham.pkl \\
      --meta outputs/overlay/<name>/overlay_meta.json \\
      --mapping outputs/mapping/<seq>/mapping_frames.csv

Chế độ overlay lấy GT từ gt_3d_scene.csv, ghép theo (frame, person).
Không được dùng a3[frame] của video nguồn vì frame video overlay có src_x/src_y
riêng cho từng người.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import make_overlay_pair as mk  # noqa: E402
import overlay_autolabel as al  # noqa: E402

SMPL_HIPS = (1, 2)


def geometric_focal(K: dict) -> float:
    return float(np.sqrt(K["fx"] * K["fy"]))


def to_true_camera(P: np.ndarray, K_wham: dict, K_true: dict) -> np.ndarray:
    """Đổi tọa độ 3D WHAM sang camera GT, giữ pixel và đổi tỉ lệ độ sâu.

    P và kết quả có cùng đơn vị (WHAM verts thường là mét nếu đầu vào WHAM
    lưu mét; hàm không tự đổi đơn vị). Trong chương trình này P được nhân
    1000 trước khi gọi nên kết quả là mm.
    """
    z = P[..., 2] * geometric_focal(K_true) / geometric_focal(K_wham)
    uv = mk.project(P, K_wham)
    return np.stack(
        [
            (uv[..., 0] - K_true["cx"]) * z / K_true["fx"],
            (uv[..., 1] - K_true["cy"]) * z / K_true["fy"],
            z,
        ],
        axis=-1,
    )


def wham_hip_positions(results: dict, J: np.ndarray) -> dict:
    """Trả về {(frame, body_id): midpoint hai hông SMPL, đơn vị mm}."""
    out = {}
    for sid, data in results.items():
        frame_ids = np.asarray(data.get("frame_ids", [])).astype(int)
        if len(frame_ids) == 0:
            continue

        verts = np.asarray(data["verts"], dtype=float)
        joints = np.einsum("jv,nvc->njc", J, verts)
        hips = joints[:, list(SMPL_HIPS)].mean(axis=1) * 1000.0

        for frame, point in zip(frame_ids, hips):
            out[(int(frame), int(sid))] = point
    return out


def load_mapping(path: Path) -> dict:
    """Đọc mapping_frames.csv thành {(frame, body_id): person}."""
    df = pd.read_csv(path, dtype={"person": str})
    required = {"frame", "body_id", "person"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} thiếu cột bắt buộc: {', '.join(sorted(missing))}"
        )

    df = df.dropna(subset=["person"]).copy()
    df["person"] = df["person"].astype(str).str.strip()
    df = df[df["person"] != ""]

    return {
        (int(row.frame), int(row.body_id)): row.person
        for row in df.itertuples(index=False)
    }


def load_overlay_gt(meta_path: Path) -> tuple[dict, dict]:
    """Đọc midpoint hai hông từ gt_3d_scene.csv và intrinsics scene_camera.

    Trả về:
      gt[(frame, person)] = pelvis giữa hai hông, mm
      K_scene = intrinsics camera cảnh, đã tính out_scale
    """
    meta_path = meta_path.resolve()
    meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
    gt_info = meta.get("gt", {})
    scene_camera = gt_info.get("scene_camera")
    if not isinstance(scene_camera, dict):
        raise SystemExit(
            "overlay_meta.json chưa có gt.scene_camera. "
            "Hãy chạy lại make_overlay_pair.py để export GT."
        )

    gt_path = meta_path.parent / "gt_3d_scene.csv"
    if not gt_path.is_file():
        raise SystemExit(f"Không tìm thấy GT scene: {gt_path}")

    df = pd.read_csv(gt_path, dtype={"person": str})
    required = {
        "frame", "person",
        "left_hip_x", "left_hip_y", "left_hip_z",
        "right_hip_x", "right_hip_y", "right_hip_z",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{gt_path} thiếu cột: {', '.join(sorted(missing))}"
        )

    left = df[["left_hip_x", "left_hip_y", "left_hip_z"]].to_numpy(float)
    right = df[["right_hip_x", "right_hip_y", "right_hip_z"]].to_numpy(float)
    mid_hips = (left + right) / 2.0

    gt = {
        (int(frame), str(person).strip()): point
        for frame, person, point in zip(df["frame"], df["person"], mid_hips)
    }

    try:
        K_scene = {
            key: float(scene_camera[key])
            for key in ("fx", "fy", "cx", "cy")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            "gt.scene_camera cần có fx, fy, cx, cy hợp lệ."
        ) from exc

    if not np.isfinite(list(K_scene.values())).all():
        raise SystemExit("Intrinsics scene_camera có giá trị không hữu hạn.")

    return gt, K_scene


def matched_overlay(
    results: dict,
    J: np.ndarray,
    mapping: dict,
    gt: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Ghép WHAM với GT overlay bằng frame + body_id -> person."""
    wham = wham_hip_positions(results, J)
    W, G, records = [], [], []

    for (frame, body_id), person in mapping.items():
        pred = wham.get((frame, body_id))
        truth = gt.get((frame, person))
        if pred is None or truth is None:
            continue
        if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
            continue
        W.append(pred)
        G.append(truth)
        records.append({"frame": frame, "body_id": body_id, "person": person})

    if not W:
        raise SystemExit(
            "Không có detection ghép được. Kiểm tra frame/body_id trong mapping, "
            "person trong mapping và gt_3d_scene.csv."
        )
    return np.asarray(W), np.asarray(G), records


def matched_single(
    results: dict,
    J: np.ndarray,
    a3: np.ndarray,
    mapping: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Đối chiếu video MPI đơn như script cũ: frame nguồn == frame video."""
    wham = wham_hip_positions(results, J)
    W, G = [], []

    for (frame, body_id), _person in mapping.items():
        pred = wham.get((frame, body_id))
        if pred is None or frame < 0 or frame >= len(a3):
            continue
        truth = np.asarray(a3[frame, (18, 23)], dtype=float).mean(axis=0)
        if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
            continue
        W.append(pred)
        G.append(truth)

    if not W:
        raise SystemExit("Không có detection phù hợp để đối chiếu video MPI đơn.")
    return np.asarray(W), np.asarray(G)


def report(name: str, P: np.ndarray, G: np.ndarray):
    error = P - G
    median_3d_cm = np.median(np.linalg.norm(error, axis=1)) / 10.0
    median_depth_cm = np.median(error[:, 2]) / 10.0
    p95_depth_cm = np.percentile(np.abs(error[:, 2]), 95) / 10.0
    median_xy_cm = np.median(np.linalg.norm(error[:, :2], axis=1)) / 10.0

    print(
        f"  {name:27s} sai số 3D trung vị {median_3d_cm:6.1f} cm | "
        f"độ sâu lệch trung vị {median_depth_cm:+7.1f} cm | "
        f"|lệch| p95 {p95_depth_cm:6.1f} cm | "
        f"ngang + dọc trung vị {median_xy_cm:5.1f} cm"
    )


def report_depth_ratio(P: np.ndarray, G: np.ndarray):
    valid = np.isfinite(G[:, 2]) & (np.abs(G[:, 2]) > 1e-8)
    if not valid.any():
        print("  Không thể tính tỉ lệ độ sâu: GT depth không hợp lệ.")
        return
    ratio = P[valid, 2] / G[valid, 2]
    print(
        f"  Tỉ lệ độ sâu quy đổi / GT: trung vị {np.median(ratio):.3f} | "
        f"p5-p95 {np.percentile(ratio, 5):.3f}-"
        f"{np.percentile(ratio, 95):.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kiểm tra hệ tọa độ camera WHAM so với GT MPI hoặc overlay"
    )
    parser.add_argument("--pkl", type=Path, required=True, help="PKL kết quả WHAM")
    parser.add_argument("--mapping", type=Path, required=True,
                        help="mapping_frames.csv")
    parser.add_argument("--meta", type=Path,
                        help="overlay_meta.json; bật chế độ video overlay")
    parser.add_argument("--annot", type=Path,
                        help="annot.mat; dùng cho chế độ MPI đơn")
    parser.add_argument("--camera", type=int,
                        help="camera index; dùng cùng --annot")
    parser.add_argument(
        "--smpl", type=Path,
        default=REPO.parent / "WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(2048, 2048),
                        metavar=("W", "H"),
                        help="kích thước video MPI đơn; mặc định 2048 2048")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.meta is not None and (args.annot is not None or args.camera is not None):
        raise SystemExit("Dùng --meta cho overlay; không kết hợp với --annot/--camera.")

    if args.meta is None and (args.annot is None or args.camera is None):
        raise SystemExit("Cần --meta hoặc đồng thời --annot và --camera.")

    mapping = load_mapping(args.mapping)
    J = al.MOD.load_j_regressor(args.smpl)
    results = joblib.load(args.pkl)

    if args.meta is not None:
        print("=== Kiểm tra video overlay ===")
        gt, K_true = load_overlay_gt(args.meta)
        meta = __import__("json").loads(args.meta.read_text(encoding="utf-8"))

        out_scale = float(meta.get("out_scale", 1.0))
        width0, height0 = map(int, meta["size"])
        width = int(round(width0 * out_scale)) // 2 * 2
        height = int(round(height0 * out_scale)) // 2 * 2
        f_wham = float(np.hypot(width, height))
        K_wham = {
            "fx": f_wham,
            "fy": f_wham,
            "cx": width / 2.0,
            "cy": height / 2.0,
        }

        W, G, records = matched_overlay(results, J, mapping, gt)
        print(f"  Detection đối chiếu được: {len(W)}")
        counts = pd.Series([r["person"] for r in records]).value_counts().to_dict()
        print(f"  Detection theo người: {counts}")
        print(
            f"  WHAM intrinsics heuristic: fx=fy={f_wham:.1f}, "
            f"cx={K_wham['cx']:.1f}, cy={K_wham['cy']:.1f}"
        )
        print(
            "  GT scene intrinsics: "
            + ", ".join(f"{key}={value:.2f}" for key, value in K_true.items())
        )

    else:
        print("=== Kiểm tra video MPI đơn ===")
        a2, a3 = mk.load_annot(args.annot, args.camera)
        K_true = mk.fit_intrinsics(a2, a3)
        width, height = args.size
        f_wham = float(np.hypot(width, height))
        K_wham = {
            "fx": f_wham,
            "fy": f_wham,
            "cx": width / 2.0,
            "cy": height / 2.0,
        }
        W, G = matched_single(results, J, a3, mapping)
        print(
            f"  Detection đối chiếu được: {len(W)} | "
            f"GT sâu trung vị {np.median(G[:, 2]) / 1000.0:.2f} m"
        )
        print(
            f"  WHAM intrinsics heuristic: fx=fy={f_wham:.1f}, "
            f"cx={K_wham['cx']:.1f}, cy={K_wham['cy']:.1f}"
        )
        print(
            "  GT camera intrinsics: "
            + ", ".join(f"{key}={value:.2f}" for key, value in K_true.items()
                         if key in ("fx", "fy", "cx", "cy"))
        )

    report("WHAM heuristic (chưa đổi)", W, G)
    W_true = to_true_camera(W, K_wham, K_true)
    report("Sau quy đổi camera", W_true, G)
    report_depth_ratio(W_true, G)


if __name__ == "__main__":
    main()
