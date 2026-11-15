#!/usr/bin/env python3
"""
Gợi ý rồi gán tay mapping body_id WHAM = người GT, với GT dạng CSV -> mapping.txt.

GT (--gt): thư mục video ghép của scripts/make_overlay_pair.py, hoặc một tệp CSV 3D:
  gt_3d.csv        (mặc định khi đưa thư mục) pose 3D gốc của từng người, mm
  gt_3d_scene.csv  như gt_3d.csv nhưng Y được tịnh tiến nguyên khối vào camera của ảnh ghép:
                   cùng hình dạng pose nên PA-MPJPE giống hệt; dùng thay gt_3d.csv được
  CSV một người    cột frame + khớp, không có cột person (CSV cũ từ annot.mat); --person đặt tên
  gt_2d.csv        tự đọc nếu nằm cùng thư mục: khớp 2D trên ảnh ghép -> gợi ý theo vị trí

WHAM (--wham): pkl của WHAM (đường dẫn hoặc Drive File ID), hoặc RA-WHAM.csv.

Gợi ý (sau đó người dùng sửa):
  mapping.txt   đã có ở thư mục ra (sửa tay, hoặc do scripts/overlay_autolabel.py ghi)
                -> làm điểm bắt đầu; --fresh để bỏ qua
  vi_tri_2d     cần pkl + gt_2d.csv + overlay_meta.json: luật của scripts/overlay_autolabel.py
                (khoảng cách 2D chuẩn hoá theo cỡ người, ngưỡng --tau). Không dùng PA-MPJPE
                nên không vòng lặp với chi phí của phương pháp được đánh giá.
  pa_mpjpe      dự phòng khi thiếu một trong ba thứ trên: mỗi frame ghép detection - người
                bằng Hungarian trên PA-MPJPE, chỉ nhận cặp < --max-pa mm. PA-MPJPE chính là
                chi phí pose của MC-MCF (và của B1/B2): nhãn theo nó thiên vị các phương pháp
                đó -> phải tự kiểm tra từng body_id, đừng nhận hàng loạt cho tập test.

Gán tay (cú pháp như scripts/wham_gt_mapping.py):
  - mặc định hỏi từng body_id; Enter = nhận gợi ý, x = không phải người GT,
    "1, 2@121-125 140" = mặc định người 1, frame 121-125 và 140 là người 2
  - --assign "3=X_S8" --assign "5=X_S8, Y_S8@100-250": gán sẵn, không hỏi (lặp được)
  - --yes: nhận gợi ý cho mọi body_id chưa --assign, không hỏi
  - hoặc sửa thẳng mapping.txt rồi chạy lại với --yes để ghi lại mapping_frames.csv

Đầu ra (--out, mặc định outputs/mapping/<tên>/; <tên> = --name hoặc tên thư mục GT. Trùng tên
một sequence trong manifest thì pose_assoc dùng ngay):
  mapping.txt, mapping_frames.csv   như scripts/wham_gt_mapping.py (đầu tệp ghi nguồn gợi ý
                                    và các body_id đã gán tay)
  RA-WHAM.csv                       khi --wham là pkl và chưa có
  frame_bodyid_scores.csv           frame, body_id, person, pa_mpjpe_mm, e2d
  subject_id_metrics.csv            trung vị PA-MPJPE / e2d theo (body_id, người)
  match_review.csv                  detection nên xem lại (sát ngưỡng, hai người gần bằng
                                    nhau, frame che nặng); chỉ có với gợi ý vi_tri_2d

Chạy từ thư mục gốc repo:
  python3 scripts/match_wham_subjects_to_gt.py \\
      --gt outputs/overlay/ovl_S8c0f2000_S8c0f3250_s0.5 \\
      --wham ../WHAM/output/demo/S8c0f2000_S8c0f3250_cam0/wham_S8c0f2000_S8c0f3250_cam0.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import overlay_autolabel as al  # noqa: E402
from pose_assoc.costs import procrustes  # noqa: E402

MOD = al.MOD
LIMBS = MOD.LIMBS
NOT_PERSON = MOD.NOT_PERSON
DEFAULT_MAX_PA_MM = 150.0
# csv_gt_poses trả khớp theo MPI_IDX = neck + 12 khớp chi đúng thứ tự LIMBS
assert list(MOD.MPI_IDX)[1:] == LIMBS

# ============================================================
# 1. ĐỌC GT VÀ WHAM
# ============================================================


def resolve_gt(arg: Path) -> tuple[Path, Optional[Path], Optional[Path]]:
    """--gt -> (CSV 3D, gt_2d.csv hoặc None, overlay_meta.json hoặc None)."""
    arg = Path(arg)
    if arg.is_dir():
        csv3d = next((arg / n for n in ("gt_3d.csv", "gt_3d_scene.csv") if (arg / n).is_file()),
                     arg / "gt_3d.csv")
    else:
        csv3d = arg
    if csv3d.name == "gt_2d.csv":
        raise SystemExit("gt_2d.csv chỉ có 2D: đưa thư mục video ghép hoặc gt_3d.csv vào --gt "
                         "(gt_2d.csv cùng thư mục sẽ tự được đọc).")
    if not csv3d.is_file():
        raise SystemExit(f"Không thấy GT 3D: {csv3d}")
    gt2d, meta = csv3d.parent / "gt_2d.csv", csv3d.parent / "overlay_meta.json"
    return csv3d, (gt2d if gt2d.is_file() else None), (meta if meta.is_file() else None)


def read_gt_2d(path: Path) -> dict:
    """gt_2d.csv -> {frame: {người: (12, 2) px}}, 12 khớp chi theo LIMBS."""
    df = pd.read_csv(path)
    cols = [f"{j}_{a}" for j in LIMBS for a in "uv"]
    missing = [c for c in ["frame", "person", *cols] if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name} thiếu cột: {', '.join(missing[:6])}")
    pts = df[cols].to_numpy(float).reshape(len(df), len(LIMBS), 2)
    out: dict = {}
    for f, who, p in zip(df["frame"].astype(int), df["person"].astype(str), pts):
        if np.isfinite(p).all():
            out.setdefault(int(f), {})[who] = p
    return out


def default_smpl() -> Path:
    return MOD.load_manifest(MOD.DEFAULT_MANIFEST)["smpl_model"]


def load_pkl(arg: str):
    """Đường dẫn pkl hoặc Drive File ID -> kết quả WHAM (bản tải tạm được xoá ngay)."""
    path = Path(arg)
    if path.is_file():
        return joblib.load(path)
    if not MOD.is_google_drive_file_id(arg):
        raise SystemExit(f"Không thấy pkl WHAM: {arg}")
    with tempfile.NamedTemporaryFile(prefix="wham_drive_", suffix=".pkl", delete=False) as f:
        tmp = Path(f.name)
    try:
        MOD.download_drive_file(arg, tmp)
        return joblib.load(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def load_wham(arg: str, smpl: Optional[Path]):
    """-> (bảng RA-WHAM, kết quả pkl hoặc None, J_regressor hoặc None)."""
    if arg.lower().endswith(".csv"):
        df = pd.read_csv(arg)
        need = ["frame_id", "body_id", *[f"{j}_{a}" for j in LIMBS for a in "xyz"]]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"{Path(arg).name} thiếu cột: {', '.join(missing[:6])}")
        return df, None, None
    results = load_pkl(arg)
    J = MOD.load_j_regressor(smpl or default_smpl())
    tables = [MOD.wham_subject_table(sid, d, J) for sid, d in results.items()
              if len(d.get("frame_ids", []))]
    if not tables:
        raise SystemExit(f"pkl không có track WHAM nào: {arg}")
    df = pd.concat(tables, ignore_index=True).sort_values(["frame_id", "body_id"])
    return df.reset_index(drop=True), results, J

# ============================================================
# 2. ĐIỂM VÀ GỢI Ý
# ============================================================


def pa_scores(wham: pd.DataFrame, gt: dict) -> pd.DataFrame:
    """PA-MPJPE (mm, 12 khớp chi) của mọi detection với mọi người GT cùng frame."""
    cols = [f"{j}_{a}" for j in LIMBS for a in "xyz"]
    pts = wham[cols].to_numpy(float).reshape(len(wham), len(LIMBS), 3)
    rows = []
    for f, bid, p in zip(wham["frame_id"].astype(int), wham["body_id"].astype(int), pts):
        for who, g in gt.get(int(f), {}).items():
            err, _ = procrustes(p, g[1:])
            rows.append((int(f), int(bid), who, err * 1000.0))
    return pd.DataFrame(rows, columns=["frame", "body_id", "person", "pa_mpjpe_mm"])


def e2d_scores(dets: pd.DataFrame, gt2d: dict) -> pd.DataFrame:
    """Sai số 2D trung bình 12 khớp / cỡ người (như overlay_autolabel) với mọi người GT."""
    rows = []
    for f, bid, p in zip(dets["frame"].astype(int), dets["body_id"].astype(int),
                         al.det_points(dets)):
        for who, g in gt2d.get(int(f), {}).items():
            e = float(np.linalg.norm(p - g, axis=1).mean() / al.person_scale(g))
            rows.append((int(f), int(bid), who, e))
    return pd.DataFrame(rows, columns=["frame", "body_id", "person", "e2d"])


def suggest_pa(scores: pd.DataFrame, tracks: dict, max_pa: float) -> dict:
    """Mỗi frame: Hungarian detection - người trên PA-MPJPE, chỉ nhận cặp < max_pa mm."""
    labels = {bid: {int(f): NOT_PERSON for f in fr} for bid, fr in tracks.items()}
    for f, g in scores.groupby("frame"):
        piv = g.pivot(index="body_id", columns="person", values="pa_mpjpe_mm")
        cost = piv.to_numpy(float)
        ok = np.isfinite(cost) & (cost < max_pa)
        rows, cols = linear_sum_assignment(np.where(ok, cost, 1e9))
        for i, j in zip(rows, cols):
            if ok[i, j]:
                labels[int(piv.index[i])][int(f)] = str(piv.columns[j])
    return labels


def suggest_2d(dets: pd.DataFrame, gt2d: dict, meta: Optional[Path], tau: float,
               margin: float) -> pd.DataFrame:
    """Luật của overlay_autolabel.label_detections; thêm mức che nếu có overlay_frames.csv."""
    lab = al.label_detections(dets, gt2d, tau, margin)
    if meta is not None:
        m = json.loads(meta.read_text(encoding="utf-8"))
        if (meta.parent / m.get("frames_csv", "overlay_frames.csv")).is_file():
            lab = al.add_occlusion(lab, meta)
    return lab


def complete(mapping: dict, tracks: dict) -> dict:
    """Đủ mọi frame của mọi track (frame thiếu = không phải người GT), đúng thứ tự track."""
    return {bid: {int(f): mapping.get(bid, {}).get(int(f), NOT_PERSON) for f in fr}
            for bid, fr in tracks.items()}


def n_diff(a: dict, b: dict) -> int:
    return sum(a[bid][f] != b[bid][f] for bid in a for f in a[bid])

# ============================================================
# 3. GÁN TAY
# ============================================================


def describe(labels: dict) -> str:
    segs = MOD.segments(labels)
    if len(segs) == 1:
        return f"{segs[0][2] or '(không phải người GT)'} (cả track)"
    head = ", ".join(f"{who or 'x'}@{a}-{b}" if a != b else f"{who or 'x'}@{a}"
                     for a, b, who, _ in segs[:4])
    return head + (f", ... ({len(segs)} đoạn)" if len(segs) > 4 else "")


def evidence(bid, scores: pd.DataFrame, review: pd.DataFrame) -> str:
    g = scores[scores["body_id"] == bid]
    parts = []
    if len(g):
        med = g.groupby("person")["pa_mpjpe_mm"].median()
        parts.append("PA-MPJPE trung vị (mm): " + " | ".join(f"{k} {v:.0f}" for k, v in med.items()))
        if "e2d" in g and g["e2d"].notna().any():
            med = g.groupby("person")["e2d"].median()
            parts.append("e2D: " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()))
    n_rev = int((review["body_id"] == bid).sum()) if len(review) else 0
    if n_rev:
        parts.append(f"xem lại: {n_rev}")
    return "    ".join(parts) or "(không có GT ở các frame của track)"


def print_help(names: list[str], source: str):
    people = "   ".join(f"{i}. {n}" for i, n in enumerate(names, 1))
    print(f"\n  Người trong GT:   {people}   x. không phải người   (gợi ý: {source})")
    print("  Enter              nhận gợi ý")
    print("  1  /  X_S8         cả track là người đó;  x = không phải người GT")
    print("  1, 2@121-125 140   mặc định người 1; frame 121-125 và 140 là người 2")
    print("  [0-2]->2, [3-11]->1  liệt kê từng đoạn")
    print("  Lệnh: v = mở video, b = quay lại, q = thoát không lưu")


def ask_body(bid, track, names, start: dict, info: str, video: Optional[Path]):
    """-> {frame: người}, hoặc 'b' / 'q', hoặc None (hỏi lại)."""
    print(f"\n  body_id {bid}: frame {track[0]}-{track[-1]} ({len(track)} detection)")
    print(f"    gợi ý: {describe(start)}")
    print(f"    {info}")
    raw = input(f"  body_id {bid} -> ").strip()
    if raw == "":
        return start
    if raw.lower() in ("b", "q"):
        return raw.lower()
    if raw.lower() == "v":
        if video is not None and video.is_file():
            MOD.open_video(video)
        else:
            print("    Không có video (thêm --video).")
        return None
    try:
        labels = MOD.resolve_labels(track, *MOD.parse_assignment(raw, names))
    except ValueError as e:
        print(f"    ✗ {e}. Nhập lại.")
        return None
    if len(set(labels.values())) > 1:
        MOD.show_segments(labels)
        if input("    Đúng chưa? [Y/n] -> ").strip().lower() not in ("", "y", "yes", "c", "co"):
            return None
    return labels


def ask_all(ids, tracks, names, start, info, video):
    """Hỏi lần lượt; -> {body_id: {frame: người}} hoặc None nếu thoát."""
    out, i = {}, 0
    while i < len(ids):
        bid = ids[i]
        try:
            ans = ask_body(bid, tracks[bid], names, start[bid], info[bid], video)
        except EOFError:
            return None
        if ans == "q":
            return None
        if ans == "b":
            i = max(0, i - 1)
        elif isinstance(ans, dict):
            out[bid] = ans
            i += 1
    return out


def parse_assign(items, tracks: dict, names: list[str]) -> dict:
    out = {}
    for item in items:
        bid, sep, spec = item.partition("=")
        try:
            bid = int(bid.strip())
        except ValueError:
            raise SystemExit(f"--assign '{item}': cần dạng BODY_ID=gán, ví dụ 3=X_S8") from None
        if not sep or bid not in tracks:
            raise SystemExit(f"--assign '{item}': body_id {bid} không có trong WHAM "
                             f"(có: {sorted(tracks)})")
        try:
            out[bid] = MOD.resolve_labels(tracks[bid], *MOD.parse_assignment(spec, names))
        except ValueError as e:
            raise SystemExit(f"--assign '{item}': {e}") from None
    return out

# ============================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ============================================================


def default_name(gt_arg: Path, csv3d: Path) -> str:
    if Path(gt_arg).is_dir():
        return Path(gt_arg).resolve().name
    if csv3d.name in ("gt_3d.csv", "gt_3d_scene.csv"):
        return csv3d.resolve().parent.name
    return csv3d.stem


def default_video(meta: Optional[Path]) -> Optional[Path]:
    if meta is None:
        return None
    v = meta.parent / json.loads(meta.read_text(encoding="utf-8")).get("video", "composite.mp4")
    return v if v.is_file() else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gợi ý + gán tay mapping body_id WHAM = người GT (GT dạng CSV)")
    p.add_argument("--gt", type=Path, required=True,
                   help="thư mục video ghép, gt_3d.csv, gt_3d_scene.csv hoặc CSV một người")
    p.add_argument("--wham", required=True, help="pkl WHAM (đường dẫn / Drive File ID) hoặc RA-WHAM.csv")
    p.add_argument("--name", help="tên sequence (mặc định: tên thư mục GT)")
    p.add_argument("--out", type=Path, help="thư mục ra (mặc định outputs/mapping/<tên>)")
    p.add_argument("--person", default="GT", help="tên người cho CSV một người (không có cột person)")
    p.add_argument("--smpl", type=Path, help="SMPL_NEUTRAL.pkl (mặc định lấy từ configs/splits.json)")
    p.add_argument("--tau", type=float, default=0.25, help="ngưỡng sai số 2D / cỡ người (vi_tri_2d)")
    p.add_argument("--margin", type=float, default=0.05, help="sát ngưỡng -> cần xem lại")
    p.add_argument("--max-pa", type=float, default=DEFAULT_MAX_PA_MM,
                   help="ngưỡng PA-MPJPE (mm) cho gợi ý dự phòng pa_mpjpe")
    p.add_argument("--assign", action="append", default=[], metavar="ID=GÁN",
                   help='gán tay không cần hỏi, ví dụ "5=X_S8, Y_S8@100-250" (lặp được)')
    p.add_argument("--yes", action="store_true", help="nhận gợi ý cho mọi body_id chưa --assign")
    p.add_argument("--fresh", action="store_true", help="bỏ qua mapping.txt đã có ở thư mục ra")
    p.add_argument("--video", type=Path, help="video mở bằng lệnh v (mặc định video ghép)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    csv3d, gt2d_path, meta = resolve_gt(args.gt)
    name = args.name or default_name(args.gt, csv3d)
    out = args.out or REPO / "outputs" / "mapping" / name
    print(f"=== {name} ===\n  GT 3D: {csv3d}")
    gt = MOD.csv_gt_poses(csv3d, args.person)
    names = MOD.gt_names(gt)
    print(f"  {len(gt)} frame GT, người: {', '.join(names)}")
    wham, results, J = load_wham(args.wham, args.smpl)
    tracks = MOD.track_frames(wham)
    print(f"  WHAM: {len(wham)} detection, {len(tracks)} body_id")

    scores = pa_scores(wham, gt)
    by_pa = suggest_pa(scores, tracks, args.max_pa)
    review = pd.DataFrame(columns=["frame", "body_id"])
    if results is not None and gt2d_path is not None and meta is not None:
        dets = al.wham_2d(results, J, al.default_intrinsics(meta))
        gt2d = read_gt_2d(gt2d_path)
        lab = suggest_2d(dets, gt2d, meta, args.tau, args.margin)
        review = lab[lab["review"]]
        scores = scores.merge(e2d_scores(dets, gt2d), on=["frame", "body_id", "person"],
                              how="outer")
        auto, source = complete(al.to_mapping(lab), tracks), f"vi_tri_2d (tau {args.tau})"
        print(f"  PA-MPJPE và vị trí 2D gợi ý khác nhau ở {n_diff(auto, by_pa)}/{len(wham)} "
              "detection")
    else:
        why = [w for w, ok in (("pkl WHAM", results is not None), ("gt_2d.csv", gt2d_path),
                               ("overlay_meta.json", meta)) if not ok]
        auto, source = by_pa, f"pa_mpjpe (< {args.max_pa:.0f} mm)"
        print(f"  ⚠ Thiếu {', '.join(why)} -> gợi ý theo PA-MPJPE. Đây là chi phí pose của "
              "MC-MCF: kiểm tra tay từng body_id, đừng nhận hàng loạt cho tập test.")

    existing = out / "mapping.txt"
    if existing.is_file() and not args.fresh:
        start, start_src = complete(MOD.read_mapping(existing, tracks, names), tracks), "mapping.txt"
        print(f"  Bắt đầu từ {existing} (khác gợi ý tự động {n_diff(start, auto)} detection; "
              "--fresh để bỏ qua)")
    else:
        start, start_src = auto, source

    manual = parse_assign(args.assign, tracks, names)
    final = {**start, **manual}
    pending = [b for b in tracks if b not in manual]
    if pending and not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("Không có terminal để hỏi: thêm --yes (nhận gợi ý cho body_id còn "
                             "lại) hoặc --assign.")
        info = {b: evidence(b, scores, review) for b in pending}
        print_help(names, start_src)
        answers = ask_all(pending, tracks, names, start, info, args.video or default_video(meta))
        if answers is None:
            print("\nĐã thoát, không lưu.")
            return 1
        final.update(answers)
        manual.update({b: v for b, v in answers.items() if v != start[b]})

    out.mkdir(parents=True, exist_ok=True)
    if results is not None and not (out / "RA-WHAM.csv").is_file():
        wham.to_csv(out / "RA-WHAM.csv", index=False)
        print(f"  Đã lưu {out / 'RA-WHAM.csv'}")
    scores.sort_values(["frame", "body_id", "person"]).to_csv(
        out / "frame_bodyid_scores.csv", index=False, float_format="%.4f")
    agg = {"pa_mpjpe_mm": "median", **({"e2d": "median"} if "e2d" in scores else {})}
    scores.groupby(["body_id", "person"]).agg(agg).reset_index().to_csv(
        out / "subject_id_metrics.csv", index=False, float_format="%.4f")
    if len(review):
        review.to_csv(out / "match_review.csv", index=False, float_format="%.4f")

    final = {b: final[b] for b in tracks}
    notes = [f"gợi ý: {source}; bắt đầu từ: {start_src}",
             f"gán tay: body_id {sorted(manual) if manual else '(không)'}; "
             f"khác gợi ý tự động: {n_diff(final, auto)}/{len(wham)} detection"]
    if source.startswith("pa_mpjpe"):
        notes.append("CẢNH BÁO: gợi ý theo PA-MPJPE (chi phí của MC-MCF) - kiểm tra tay trước "
                     "khi dùng cho tập test")
    MOD.write_mapping(out, final, {"name": name, "wham_pkl": args.wham, "gt": {"path": csv3d}},
                      notes)
    counts = pd.Series([w for lab in final.values() for w in lab.values()]).value_counts()
    print("  Kết quả: " + ", ".join(f"{k or '(không phải người GT)'} {v}"
                                    for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nĐã dừng (Ctrl+C).")
