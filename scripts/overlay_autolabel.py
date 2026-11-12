#!/usr/bin/env python3
"""
Gán nhãn tự động body_id WHAM ↔ người GT cho video ghép (gt.type = "mpi_overlay",
tạo bằng scripts/make_overlay_pair.py). Kết quả cùng định dạng với gán tay.

Nhãn dựa trên VỊ TRÍ 2D, không dùng PA-MPJPE (là chi phí của chính phương pháp được
đánh giá; dùng nó để gán nhãn sẽ thành vòng lặp):
  - khớp 3D hệ camera của WHAM (verts trong pkl) -> chiếu 2D bằng intrinsics MẠNG WHAM
    đã dùng: heuristic f = sqrt(W² + H²), điểm chính = tâm ảnh. Bản WHAM ở ../WHAM luôn
    dùng heuristic này cho verts/trans (lib/data/datasets/dataset_custom.py), kể cả khi
    chạy với --calib: calib chỉ vào DPVO/SLAM. Đã kiểm trên H4D cam07 (calib fx = 1758.7):
    verts chỉ khớp keypoint ViTPose khi chiếu bằng f = 4405.7. Vì vậy đừng truyền --calib
    trừ khi dùng bản WHAM khác có đưa calib vào mạng.
  - GT 2D: annot2 của X; annot2 của Y thu nhỏ quanh tâm ảnh (overlay_meta.json);
    nhân out_scale của video
  - e_k = khoảng cách trung bình 12 khớp tới người k / cỡ người k trên ảnh
  - nhãn = argmin_k e_k nếu e_min < tau; vùng tau..tau_hi chỉ nhận khi cùng track có
    frame chắc chắn là người đó trong ±window frame; còn lại "không phải người GT"
  - sát ngưỡng hoặc hai người gần bằng nhau -> autolabel_review.csv để xem lại bằng tay

Kiểm chứng trên S8 cam0 (1 người, đã gán tay, WHAM chạy với intrinsics mặc định):
detection đúng có e trung vị 0.04; detection không phải S8 có e >= 0.26; nhãn tự động
trùng nhãn tay 9107/9107 detection (170 dòng được đánh dấu xem lại).

Đầu ra (outputs/mapping/<seq>/):
  mapping.txt, mapping_frames.csv   như gán tay -> dùng ngay cho pose_assoc
  RA-WHAM.csv                       nếu chưa có
  wham_2d.csv                       12 khớp 2D của mọi detection WHAM
  autolabel_review.csv              detection sát ngưỡng hoặc ở frame che nặng: xem lại
  autolabel_audit.csv               mẫu phân tầng theo mức che: điền manual_person bằng tay

Chạy từ thư mục gốc repo:
  python3 scripts/overlay_autolabel.py --seq ovl_S8c0f2000_S8c0f3250_s0.5
  python3 scripts/overlay_autolabel.py --audit-score outputs/mapping/<seq>/autolabel_audit.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

REPO = Path(__file__).resolve().parent.parent


def mapping_module():
    spec = importlib.util.spec_from_file_location("wham_gt_mapping",
                                                  REPO / "scripts" / "wham_gt_mapping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = mapping_module()
LIMBS = MOD.LIMBS
MPI_LIMB_IDX = [MOD.MPI_IDX[j] for j in LIMBS]
WHAM_LIMB_IDX = [MOD.WHAM_SMPL_IDX[j] for j in LIMBS]

# ============================================================
# 1. 2D CỦA WHAM VÀ CỦA GT
# ============================================================


def read_calib(path: Path) -> tuple[float, float, float, float]:
    vals = [float(x) for x in path.read_text().split()[:4]]
    if len(vals) != 4:
        raise SystemExit(f"calib cần 4 số 'fx fy cx cy': {path}")
    return tuple(vals)


def wham_2d(results: dict, J: np.ndarray, K) -> pd.DataFrame:
    """Khớp 12 chi của mọi detection WHAM chiếu lên ảnh: cột frame, body_id, <khớp>_u/_v."""
    fx, fy, cx, cy = K
    tables = []
    for sid, d in results.items():
        if not len(d.get("frame_ids", [])):
            continue
        jc = np.einsum("jv,nvc->njc", J, np.asarray(d["verts"], float))[:, WHAM_LIMB_IDX]
        z = np.maximum(jc[..., 2], 1e-6)
        u, v = fx * jc[..., 0] / z + cx, fy * jc[..., 1] / z + cy
        df = pd.DataFrame({"frame": np.asarray(d["frame_ids"]).astype(int), "body_id": int(sid)})
        for i, name in enumerate(LIMBS):
            df[f"{name}_u"], df[f"{name}_v"] = u[:, i], v[:, i]
        tables.append(df)
    return pd.concat(tables, ignore_index=True).sort_values(["frame", "body_id"])


def det_points(df: pd.DataFrame) -> np.ndarray:
    cols = [c for j in LIMBS for c in (f"{j}_u", f"{j}_v")]
    return df[cols].to_numpy(float).reshape(len(df), len(LIMBS), 2)


def overlay_gt_2d(meta_path: Path) -> dict:
    """{frame: {người: (12, 2)}} trên ảnh ghép: tâm + scale * (annot2 - tâm), nhân out_scale."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base, out_scale = meta_path.parent, float(meta.get("out_scale", 1.0))
    frames = pd.read_csv(base / meta["frames_csv"])
    out: dict = {}
    for name, p in meta["persons"].items():
        data = loadmat((base / p["annot"]).resolve(), squeeze_me=True, struct_as_record=False)
        a2 = np.asarray(data["annot2"][int(p["camera"])], float)
        a2 = a2.reshape(len(a2), -1, 2)[:, MPI_LIMB_IDX]
        s, c = float(p.get("scale", 1.0)), np.asarray(p.get("center", (0.0, 0.0)), float)
        for f, src in zip(frames["frame"].to_numpy(), frames[p["src_col"]].to_numpy()):
            out.setdefault(int(f), {})[name] = (c + s * (a2[src] - c)) * out_scale
    return out


def default_intrinsics(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    w, h = (np.asarray(meta["size"], float) * float(meta.get("out_scale", 1.0))).round()
    f = float(np.hypot(w, h))
    return f, f, w / 2.0, h / 2.0

# ============================================================
# 2. LUẬT GÁN NHÃN
# ============================================================


def person_scale(pts: np.ndarray) -> float:
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 40.0))


def label_detections(dets: pd.DataFrame, gt2d: dict, tau: float = 0.25,
                     margin: float = 0.05, tau_hi: float = 0.5,
                     window: int = 25) -> pd.DataFrame:
    """Mỗi detection -> người gần nhất trên ảnh (chuẩn hoá theo cỡ người).

      e_min < tau                 : người gần nhất (chắc chắn)
      tau <= e_min < tau_hi       : vùng không chắc. Chỉ nhận người gần nhất nếu cùng
                                    track có frame chắc chắn là người đó trong ±window
                                    frame (WHAM đặt sai vị trí lúc đầu/cuối track); còn lại
                                    coi là không phải người GT
      e_min >= tau_hi             : không phải người GT

    Trả về bảng: frame, body_id, person ("" = không phải người GT), best, e_best,
    second, e_second, review (True nếu sát ngưỡng, nhận nhờ track, hoặc hai người gần
    như bằng nhau).
    """
    pts = det_points(dets)
    rows = []
    for (f, bid), p in zip(zip(dets["frame"].to_numpy(), dets["body_id"].to_numpy()), pts):
        people = gt2d.get(int(f), {})
        errs = sorted((float(np.linalg.norm(p - g, axis=1).mean() / person_scale(g)), k)
                      for k, g in people.items())
        e1, k1 = errs[0] if errs else (np.inf, "")
        e2, k2 = errs[1] if len(errs) > 1 else (np.inf, "")
        rows.append((int(f), int(bid), k1 if e1 < tau else "", k1, e1, k2, e2))
    lab = pd.DataFrame(rows, columns=["frame", "body_id", "person", "best", "e_best",
                                      "second", "e_second"])
    lab["review"] = (lab["e_best"] - tau).abs() < margin
    lab["review"] |= (lab["person"] != "") & (lab["e_second"] - lab["e_best"] < margin)
    unsure = (lab["e_best"] >= tau) & (lab["e_best"] < tau_hi)
    for _, g in lab[lab["e_best"] < tau_hi].groupby("body_id"):
        sure = g[g["e_best"] < tau]
        for i in g.index[unsure[g.index]]:
            near = sure[(sure["frame"] - lab.at[i, "frame"]).abs() <= window]
            if (near["person"] == lab.at[i, "best"]).any():
                lab.at[i, "person"] = lab.at[i, "best"]
                lab.at[i, "review"] = True
    return lab


def to_mapping(labels: pd.DataFrame) -> dict:
    """-> {body_id: {frame: người}} như scripts/wham_gt_mapping.py ghi ra."""
    return {int(b): dict(zip(g["frame"].astype(int), g["person"]))
            for b, g in labels.sort_values(["body_id", "frame"]).groupby("body_id")}

# ============================================================
# 3. MỨC CHE VÀ KIỂM CHỨNG BẰNG TAY THEO MỨC CHE
# ============================================================
# Đúng 9107/9107 trên video MỘT người chưa chứng minh được gì cho lúc hai người chồng nhau:
# khi bị che, WHAM có thể "bịa" khớp bị che nên khoảng cách 2D kém tin cậy. Vì vậy:
#   - mỗi detection mang mức che của frame (vis_y: phần người đứng sau còn thấy);
#   - detection ở frame che nặng luôn bị đánh dấu xem lại;
#   - autolabel_audit.csv: mẫu ngẫu nhiên phân tầng theo mức che để gán tay cột manual_person,
#     rồi --audit-score báo độ đúng của nhãn tự động theo từng mức che.

OCCL_BINS = [(-0.01, 0.5, "che_nang"), (0.5, 0.9, "che_vua"), (0.9, 1.01, "khong_che")]


def add_occlusion(labels: pd.DataFrame, meta_path: Path, heavy: float = 0.5) -> pd.DataFrame:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = pd.read_csv(meta_path.parent / meta["frames_csv"])[["frame", "vis_y"]]
    lab = labels.drop(columns=[c for c in ("vis_y", "occlusion") if c in labels]).merge(
        frames, on="frame", how="left")
    lab["vis_y"] = lab["vis_y"].fillna(1.0)
    lab["occlusion"] = ""
    for lo, hi, name in OCCL_BINS:
        lab.loc[(lab["vis_y"] > lo) & (lab["vis_y"] <= hi), "occlusion"] = name
    lab.loc[lab["vis_y"] < heavy, "review"] = True
    return lab


def audit_sample(labels: pd.DataFrame, per_bin: int = 40, seed: int = 0) -> pd.DataFrame:
    """Mẫu phân tầng theo mức che (mỗi mức tối đa per_bin detection) để gán tay."""
    parts = [g.sample(min(per_bin, len(g)), random_state=seed)
             for _, g in labels.groupby("occlusion")]
    out = pd.concat(parts)[["frame", "body_id", "occlusion", "vis_y", "person", "e_best"]]
    out = out.rename(columns={"person": "auto_person"}).sort_values(["occlusion", "frame"])
    out["manual_person"] = ""          # điền: tên người (X_..., Y_...) hoặc "x" = không phải người
    return out


def audit_score(path: Path) -> pd.DataFrame:
    a = pd.read_csv(path, dtype={"auto_person": str, "manual_person": str}).fillna("")
    a = a[a["manual_person"].str.strip() != ""].copy()
    if a.empty:
        raise SystemExit(f"Chưa có dòng nào điền manual_person trong {path}.")
    manual = a["manual_person"].str.strip().replace({"x": "", "X": "", "-": ""})
    a["correct"] = manual == a["auto_person"]
    res = a.groupby("occlusion")["correct"].agg(["count", "sum", "mean"]).rename(
        columns={"count": "n", "sum": "dung", "mean": "do_dung"})
    res.loc["tong"] = [len(a), int(a["correct"].sum()), a["correct"].mean()]
    return res

# ============================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ============================================================


def find_sequence(manifest: dict, name: str) -> dict:
    for split in MOD.SPLITS:
        for seq in manifest.get(split, []):
            if seq["name"] == name:
                return seq
    raise SystemExit(f"Không có sequence '{name}' trong manifest.")


def load_results(seq: dict):
    """-> (kết quả WHAM, đường dẫn pkl tạm cần xoá hoặc None)."""
    if not seq.get("wham_pkl_is_drive"):
        print(f"  Đọc pkl WHAM: {seq['wham_pkl']}")
        return joblib.load(seq["wham_pkl"]), None
    with tempfile.NamedTemporaryFile(prefix="wham_drive_", suffix=".pkl", delete=False) as f:
        tmp = Path(f.name)
    MOD.download_drive_file(seq["wham_pkl"], tmp)
    return joblib.load(tmp), tmp


def write_ra_wham(results: dict, J: np.ndarray, out_csv: Path):
    if out_csv.is_file():
        return
    tables = [MOD.wham_subject_table(sid, d, J) for sid, d in results.items()
              if len(d.get("frame_ids", []))]
    pd.concat(tables, ignore_index=True).sort_values(["frame_id", "body_id"]).to_csv(
        out_csv, index=False)
    print(f"  Đã lưu {out_csv.name}")


def summarize(labels: pd.DataFrame):
    used = labels[labels["person"] != ""]
    print(f"  {len(labels)} detection: " + ", ".join(
        f"{k or '(không phải người GT)'} {v}" for k, v in labels["person"].value_counts().items()))
    if len(used):
        med = used["e_best"].median()
        print(f"  Sai số 2D trung vị của detection được gán: {med:.3f} cỡ người")
        if med > 0.12:
            print("  ⚠ Sai số lớn bất thường: kiểm tra intrinsics mạng WHAM đã dùng "
                  "(bản ../WHAM: heuristic, không truyền --calib).")
    if "occlusion" in labels:
        by = labels.groupby("occlusion").agg(n=("frame", "size"), xem_lai=("review", "sum"),
                                            e_trung_vi=("e_best", "median"))
        print("  Theo mức che của frame:\n" + by.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"  Cần xem lại: {int(labels['review'].sum())} detection (autolabel_review.csv)")
    print("  Độ đúng của nhãn tự động khi hai người chồng nhau CHƯA được kiểm chứng: điền cột "
          "manual_person trong autolabel_audit.csv rồi chạy --audit-score.")


def parse_args():
    p = argparse.ArgumentParser(description="Gán nhãn tự động cho video ghép mpi_overlay")
    p.add_argument("--seq", help="tên sequence trong manifest")
    p.add_argument("--audit-score", type=Path, metavar="CSV",
                   help="chấm autolabel_audit.csv đã điền manual_person, theo mức che")
    p.add_argument("--manifest", type=Path, default=REPO / "configs" / "splits.json")
    p.add_argument("--calib", type=Path,
                   help="'fx fy cx cy' mà MẠNG WHAM dùng cho verts. Bỏ trống với bản ../WHAM "
                        "(luôn dùng heuristic sqrt(W²+H²), tâm ảnh; calib.txt của WHAM chỉ cho SLAM)")
    p.add_argument("--tau", type=float, default=0.25, help="ngưỡng sai số 2D / cỡ người")
    p.add_argument("--margin", type=float, default=0.05, help="sát ngưỡng -> cần xem lại")
    p.add_argument("--force", action="store_true", help="ghi đè mapping.txt đã có")
    return p.parse_args()


def main():
    args = parse_args()
    if args.audit_score:
        print(audit_score(args.audit_score).to_string(float_format=lambda v: f"{v:.3f}"))
        return
    if not args.seq:
        raise SystemExit("Cần --seq (hoặc --audit-score CSV).")
    manifest = MOD.load_manifest(args.manifest)
    seq = find_sequence(manifest, args.seq)
    if seq["gt"]["type"] != "mpi_overlay":
        raise SystemExit("Chỉ dùng cho gt.type = 'mpi_overlay'; sequence khác hãy gán tay.")
    out = seq["out_dir"]
    if (out / "mapping.txt").is_file() and not args.force:
        raise SystemExit(f"Đã có {out / 'mapping.txt'} (--force để ghi đè).")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = seq["gt"]["path"]
    K = read_calib(args.calib) if args.calib else default_intrinsics(meta_path)
    print(f"=== {seq['name']} ===\n  intrinsics fx={K[0]:.1f} fy={K[1]:.1f} cx={K[2]:.1f} cy={K[3]:.1f}")
    results, tmp = load_results(seq)
    try:
        J = MOD.load_j_regressor(manifest["smpl_model"])
        write_ra_wham(results, J, out / "RA-WHAM.csv")
        dets = wham_2d(results, J, K)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    dets.to_csv(out / "wham_2d.csv", index=False, float_format="%.1f")
    labels = label_detections(dets, overlay_gt_2d(meta_path), args.tau, args.margin)
    labels = add_occlusion(labels, meta_path)
    labels[labels["review"]].to_csv(out / "autolabel_review.csv", index=False, float_format="%.4f")
    audit_sample(labels).to_csv(out / "autolabel_audit.csv", index=False, float_format="%.4f")
    MOD.write_mapping(out, to_mapping(labels), seq)
    summarize(labels)


if __name__ == "__main__":
    main()
