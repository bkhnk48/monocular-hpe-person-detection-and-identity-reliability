#!/usr/bin/env python3
"""
Ghép detection WHAM với người GT của Harmony4D -> file nhãn identity.

Nguyên tắc: ghép bằng 2D (IoU giữa bbox chiếu từ mesh WHAM và bbox GT của
camera), KHÔNG dùng 3D/PA-MPJPE, vì:
  - depth của WHAM sai tỉ lệ (focal giả định), sai số này chính là đối tượng nghiên cứu;
  - PA-MPJPE loại bỏ translation/scale nên không thấy lỗi vị trí.

Mesh WHAM ('verts', camera-space) được chiếu bằng đúng mô hình camera WHAM dùng
khi chạy demo không có calib: pinhole, f = sqrt(W^2 + H^2), tâm ảnh (W/2, H/2).
Video Harmony4D là ảnh fisheye; WHAM chạy trên chính ảnh đó nên bbox chiếu ra
cùng hệ tọa độ pixel với bbox GT.

Mỗi frame: gán 1-1 (Hungarian / vét cạn) giữa detection và người GT theo IoU.
  - IoU >= --iou-thresh  -> identity = tên người GT (vd. aria01)
  - còn lại               -> identity = "unmatched"
"unmatched" KHÔNG đồng nghĩa ghost: Harmony4D chỉ gán GT cho người đeo Aria,
người thật khác trong cảnh (khán giả, nhân viên) cũng sẽ rơi vào nhóm này.
Phải xem ảnh RGB trước khi gọi một track là hallucination.

Cột "ambiguous" = 1 khi detection có IoU >= ngưỡng với >= 2 người GT
(thường gặp khi hai người vật lộn chồng lên nhau) -> nên loại khỏi phân tích
nhạy cảm hoặc báo cáo riêng.

Dùng (trực tiếp từ thư mục Harmony4D):
  python scripts/label_harmony4d_wham.py \
      --wham-pkl .../wham_harmony4D_041_grappling2_cam01_ffmpeg.pkl \
      --h4d-seq-dir .../03_grappling2/041_grappling2 --cam cam01 \
      --out outputs/harmony4d/041_grappling2_cam01

Hoặc từ GT đã đóng gói (data/processed/harmony4d/gt_041_grappling2_cam01.pkl):
  python scripts/label_harmony4d_wham.py --wham-pkl ... --gt-pkl ... --out ...
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def load_wham(path: Path) -> dict:
    try:
        import joblib

        data = joblib.load(path)
    except ImportError:
        with open(path, "rb") as f:
            data = pickle.load(f)
    return {int(k): v for k, v in data.items() if isinstance(v, dict) and "verts" in v}


def _npy_dict(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    return obj.item() if obj.dtype == object and obj.shape == () else {}


def load_gt_from_seq(seq_dir: Path, cam: str) -> tuple[dict, tuple[int, int]]:
    """{gt_frame_index: {person_id: bbox[x1,y1,x2,y2]}} và (W, H) của camera."""
    pdata = seq_dir / "processed_data"
    frames = {}
    for f in sorted(glob.glob(str(pdata / "bbox" / cam / "*.npy"))):
        idx = int(Path(f).stem)
        frames[idx] = {p: np.asarray(b, float)[:4] for p, b in _npy_dict(f).items()}
    if not frames:
        raise FileNotFoundError(f"Không thấy bbox GT trong {pdata/'bbox'/cam}")
    size = _camera_size(seq_dir, cam)
    return frames, size


def _camera_size(seq_dir: Path, cam: str) -> tuple[int, int]:
    cams = seq_dir / "colmap" / "workplace" / "cameras.txt"
    cam_id = int(cam.replace("cam", ""))
    if cams.is_file():
        for line in cams.read_text().splitlines():
            parts = line.split()
            if parts and not line.startswith("#") and int(parts[0]) == cam_id:
                return int(parts[2]), int(parts[3])
    return 3840, 2160


def load_gt_from_pkl(path: Path) -> tuple[dict, tuple[int, int]]:
    with open(path, "rb") as f:
        g = pickle.load(f)
    frames = {int(k): {p: np.asarray(r["bbox"], float)[:4] for p, r in v.items()}
              for k, v in g["frames"].items()}
    line = g.get("meta", {}).get("cam01_intrinsics_line", "")
    parts = line.split()
    size = (int(parts[2]), int(parts[3])) if len(parts) > 3 else (3840, 2160)
    return frames, size


# ---------------------------------------------------------------------------
# Hình học
# ---------------------------------------------------------------------------
def project_boxes(verts: np.ndarray, focal: float, W: int, H: int) -> np.ndarray:
    """verts (N, V, 3) camera-space -> bbox (N, 4) pixel."""
    z = np.clip(verts[..., 2:3], 1e-6, None)
    uv = focal * verts[..., :2] / z + np.array([W / 2.0, H / 2.0])
    return np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / area) if area > 0 else 0.0


def best_assignment(M: np.ndarray) -> list[tuple[int, int]]:
    """Gán 1-1 tối đa tổng IoU. Dùng scipy nếu có, ngược lại vét cạn (số người nhỏ)."""
    n_d, n_g = M.shape
    if n_d == 0 or n_g == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-M)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:
        best, best_val = [], -1.0
        if n_d <= n_g:
            for perm in itertools.permutations(range(n_g), n_d):
                v = sum(M[i, perm[i]] for i in range(n_d))
                if v > best_val:
                    best, best_val = [(i, perm[i]) for i in range(n_d)], v
        else:
            for perm in itertools.permutations(range(n_d), n_g):
                v = sum(M[perm[j], j] for j in range(n_g))
                if v > best_val:
                    best, best_val = [(perm[j], j) for j in range(n_g)], v
        return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wham-pkl", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h4d-seq-dir", type=Path, help="Thư mục sequence Harmony4D, vd. 041_grappling2")
    src.add_argument("--gt-pkl", type=Path, help="GT đã đóng gói (data/processed/harmony4d/*.pkl)")
    ap.add_argument("--cam", default="cam01")
    ap.add_argument("--frame-offset", type=int, default=1,
                    help="gt_frame = wham_frame + offset (Harmony4D đánh số ảnh từ 00001)")
    ap.add_argument("--focal", type=float, default=None,
                    help="Focal WHAM đã dùng; mặc định sqrt(W^2+H^2) như WHAM demo")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wham = load_wham(args.wham_pkl)
    if args.h4d_seq_dir:
        gt, (W, H) = load_gt_from_seq(args.h4d_seq_dir, args.cam)
    else:
        gt, (W, H) = load_gt_from_pkl(args.gt_pkl)
    focal = args.focal or float(np.hypot(W, H))
    persons = sorted({p for fr in gt.values() for p in fr})

    # Bảng detection
    rows = []
    for bid, res in wham.items():
        frames = np.asarray(res["frame_ids"]).astype(int).reshape(-1)
        boxes = project_boxes(np.asarray(res["verts"], float), focal, W, H)
        for f, b in zip(frames, boxes):
            rows.append((int(f), bid, *b))
    det = pd.DataFrame(rows, columns=["frame_id", "body_id", "x1", "y1", "x2", "y2"])

    out_rows = []
    for f, g in det.groupby("frame_id"):
        gt_f = gt.get(f + args.frame_offset, {})
        pids = [p for p in persons if p in gt_f]
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        M = np.array([[iou(b, gt_f[p]) for p in pids] for b in boxes]).reshape(len(boxes), len(pids))
        assign = dict(best_assignment(M)) if len(pids) else {}
        for i, (_, r) in enumerate(g.iterrows()):
            ious = {p: (M[i, j] if len(pids) else 0.0) for j, p in enumerate(pids)}
            j = assign.get(i)
            matched = j is not None and M[i, j] >= args.iou_thresh
            out_rows.append(dict(
                frame_id=f, body_id=int(r.body_id),
                identity=pids[j] if matched else "unmatched",
                iou=round(float(M[i, j]), 4) if j is not None else 0.0,
                ambiguous=int(sum(v >= args.iou_thresh for v in ious.values()) >= 2),
                gt_present=len(pids),
                **{f"iou_{p}": round(float(ious.get(p, np.nan)), 4) for p in persons},
                x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2,
            ))
    lab = pd.DataFrame(out_rows).sort_values(["frame_id", "body_id"])
    lab.to_csv(args.out / "labels.csv", index=False)

    # Tóm tắt theo track
    tab = pd.crosstab(lab.body_id, lab.identity)
    summary = tab.copy()
    summary["n"] = tab.sum(axis=1)
    summary["dominant"] = tab.idxmax(axis=1)
    summary["purity"] = (tab.max(axis=1) / summary["n"]).round(3)
    span = lab.groupby("body_id").frame_id.agg(first="min", last="max")
    box = lab.groupby("body_id")[["x1", "y1", "x2", "y2"]].median().round(0)
    summary = summary.join(span).join(box.add_prefix("median_"))
    summary.to_csv(args.out / "track_summary.csv")

    # Độ phủ GT: frame nào người GT không có detection nào khớp
    cov = []
    for p in persons:
        frames_gt = sorted(k - args.frame_offset for k, v in gt.items() if p in v)
        hit = set(lab.loc[lab.identity == p, "frame_id"])
        cov.append(dict(person=p, gt_frames=len(frames_gt),
                        matched_frames=sum(f in hit for f in frames_gt),
                        recall=round(sum(f in hit for f in frames_gt) / max(len(frames_gt), 1), 3)))
    cov = pd.DataFrame(cov)
    cov.to_csv(args.out / "gt_coverage.csv", index=False)

    print(f"Image {W}x{H}, focal={focal:.1f}, IoU>={args.iou_thresh}, offset={args.frame_offset}")
    print("\nTrack summary:\n", summary.to_string())
    print("\nGT coverage:\n", cov.to_string(index=False))
    print(f"\nAmbiguous detections: {int(lab.ambiguous.sum())}/{len(lab)}")
    print(f"-> {args.out.resolve()}")


if __name__ == "__main__":
    main()
