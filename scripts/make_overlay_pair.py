#!/usr/bin/env python3
"""
Tạo video 2 người từ 2 video MPI-INF-3DHP 1 người, theo chiều sâu:
  X = video nền, người X ở cỡ gốc và LUÔN đứng trước.
  Y = người trong video Y (có mặt nạ), thu nhỏ hệ số s (mặc định 0.5, diện tích 1/4)
      quanh tâm ảnh rồi đặt vào khung hình của X, đứng sau X.
  Mỗi người giữ nguyên chuyển động của mình theo frame; số frame = min(độ dài X, Y).

Hình học: với cùng intrinsics K, ảnh thu nhỏ s quanh điểm chính (cx, cy) xấp xỉ ảnh của
chính thân người Y lùi ra sau: giữ toạ độ ngang và độ cao, độ sâu Z -> Z/s (KHÔNG phải lùi
theo tia nhìn: lùi theo tia nhìn thì ảnh không đổi). Sai số phối cảnh trên S8 cam0: trung vị
1.6% chiều cao người (p95 5.4%), lớn khi Y ở rất gần camera. Độ cao giữ nguyên nên với
camera gần nằm ngang chân Y vẫn trên sàn. Người xa hơn (nhỏ hơn) luôn bị X che.
Thu nhỏ quanh TÂM ẢNH: đó cũng là điểm chính mà mạng WHAM giả định (WHAM luôn dùng
f = sqrt(W² + H²), tâm ảnh cho verts/trans; calib chỉ vào SLAM), nên với WHAM phép biến
đổi này nhất quán hoàn toàn. So với camera thật (cam0 cy = 1051.4), Y cao hơn
(1 - s) * 27 ≈ 14 px; độ lệch hằng số này bù được khi biến đổi GT.

PA-MPJPE bất biến với dịch, xoay và tỉ lệ, nên GT 3D của Y vẫn là annot3 gốc.
GT 2D của Y trong video ghép = tâm + s * (annot2 - tâm); dùng cho gán nhãn tự động
(scripts/overlay_autolabel.py).

Mặt nạ = điểm ảnh không phải phông xanh (chroma key) ∩ vùng quanh khung xương 2D
(annot2). Không cần FGmasks của MPI. Độ dày vùng xương tỉ lệ với cỡ người tính theo độ sâu
(depth_scale: f * --ref-height / Z_pelvis từ annot3), không theo khung 2D: khung 2D co lại
khi ngồi xổm / cúi nên vùng xương hẹp đi và cắt mất đùi, bắp chân.

Đầu ra (outputs/overlay/<name>/):
  composite.mp4       video ghép H.264 -> chạy WHAM
  frames_raw/         (--mode generate) từng frame PNG <frame ghép:06d>.png, ~3 MB/frame ở 2048²
  frames_edit/        frame đã sửa tay: PNG cùng tên, cùng kích thước; --mode render ưu tiên
  overlay_meta.json   nguồn, phép thu nhỏ, tên người (manifest: gt.type = "mpi_overlay")
  overlay_frames.csv  mỗi frame: frame nguồn X/Y, diện tích mặt nạ, tỉ lệ Y còn thấy
  gt_3d.csv           GT 3D gốc (annot3, mm) từng người: GT cho chi phí PA-MPJPE
  gt_3d_scene.csv     GT 3D trong camera của video ghép (mm): Y lùi ra sau; cho C_root
  gt_2d.csv           GT 2D trên ảnh ghép (px): Y đã thu nhỏ; cho gán nhãn / C_2D
  preview/            frame mẫu có vẽ khung (--preview-every)

Chạy từ thư mục gốc repo:
  --x / --y có dạng START:END (END không được lấy). Hai đoạn được đọc từ video gốc.
  --x-video là video gốc của X; nếu bỏ --y-video, Y cũng đọc từ chính video đó.
  Subject/camera dùng để đọc annot.mat được suy ra từ đường dẫn video dạng
  .../<SUBJECT>/Seq1/imageSequences/video_cam<CAMERA>.avi; có thể ghi đè bằng
  --x-subject/--x-camera và --y-subject/--y-camera. Video Seq khác Seq1 cần
  --annot-pattern có {seq}. Cùng video (cùng annot.mat) thì hai đoạn không được giao nhau
  hoặc cách nhau < --min-gap frame (trừ khi --allow-overlap): phần giao là cùng một chuỗi
  động tác xuất hiện hai lần (Y là bản thu nhỏ của chính X).

  V=../WHAM/inputs/mpi_inf_3dhpe/S8/Seq1/imageSequences/video_cam0.avi
  python3 scripts/make_overlay_pair.py --x 0:6468 --y 0:6468 --x-video $V --frames 1500 --suggest
  python3 scripts/make_overlay_pair.py --x 0:1500 --y 4750:6250 --x-video $V           # -> MP4
  python3 scripts/make_overlay_pair.py --x 0:1500 --y 4750:6250 --x-video $V --mode generate
  #   ... sửa frames_raw/000123.png, lưu PNG cùng tên vào frames_edit/ ...
  python3 scripts/make_overlay_pair.py --mode render --name ovl_S8c0f0-1500_S8c0f4750-6250_s0.5
  python3 scripts/make_overlay_pair.py --x 0:1500 --y 0:1500 --x-video $V \
      --y-video ../WHAM/inputs/mpi_inf_3dhpe/S1/Seq1/imageSequences/video_cam0.avi
  python3 scripts/make_overlay_pair.py --gt-only outputs/overlay/ovl_*
  python3 scripts/make_overlay_pair.py --mask-qa outputs/overlay/ovl_*
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import binary_fill_holes, label

REPO = Path(__file__).resolve().parent.parent
VIDEO_PATTERN = "../WHAM/inputs/mpi_inf_3dhpe/{subject}/Seq1/imageSequences/video_cam{camera}.avi"
ANNOT_PATTERN = "data/raw/mpi_inf_3dhp/{subject}/annot.mat"
REF_HEIGHT_MM = 1750.0           # chiều cao tham chiếu cho depth_scale (S8: chuỗi xương ~1670 mm)
# Phông MPI: (G - max(R,B)) / G ~ 0.42-0.74 sát người. Áo len xám-xanh của S4: ~0.2 nhưng
# G - max(R,B) ~ 15-35 > tau, nên chỉ dùng tau thì mất ~56% điểm ảnh người (Y lộ ra đè lên X).
MIN_CHROMA = 0.40
FILL_WARN = 0.45                 # mặt nạ / vùng quanh khung xương; S8 ~0.6 (trung vị)

# 28 khớp MPI-INF-3DHP (zero-based)
HEAD = 6
BONES = [(7, 6), (6, 5), (5, 1), (1, 0), (0, 2), (2, 3), (3, 4),
         (5, 8), (8, 9), (9, 10), (10, 11), (11, 12),
         (5, 13), (13, 14), (14, 15), (15, 16), (16, 17),
         (4, 18), (18, 19), (19, 20), (20, 21), (21, 22),
         (4, 23), (23, 24), (24, 25), (25, 26), (26, 27)]
TORSO = [9, 14, 23, 18]          # vai trái, vai phải, hông phải, hông trái
PELVIS, NECK = 4, 5
# khớp xuất ra file GT: pelvis + 13 khớp như data/raw/mpi_inf_3dhp/ground_truth/*.csv
# (cùng chỉ số với MPI_IDX trong scripts/wham_gt_mapping.py)
GT_JOINTS = {"pelvis": 4, "neck": 5, "left_shoulder": 9, "right_shoulder": 14,
             "left_elbow": 10, "right_elbow": 15, "left_wrist": 11, "right_wrist": 16,
             "left_hip": 18, "right_hip": 23, "left_knee": 19, "right_knee": 24,
             "left_ankle": 20, "right_ankle": 25}


# ============================================================
# 1. NGUỒN
# ============================================================


def parse_segment(spec: str) -> dict:
    """Phân tích khoảng frame START:END, theo quy ước [START, END)."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise SystemExit(f"Khoảng frame '{spec}' không hợp lệ; dùng START:END, ví dụ 0:1500.")
    try:
        start, end = map(int, parts)
    except ValueError:
        raise SystemExit(f"Khoảng frame '{spec}' không hợp lệ: START và END phải là số nguyên.")
    if start < 0 or end <= start:
        raise SystemExit(f"Khoảng frame '{spec}' không hợp lệ; cần 0 <= START < END.")
    return {"start": start, "end": end}


def video_identity(path: Path, subject_override=None, camera_override=None) -> dict:
    """Suy ra subject/seq/camera từ .../<SUBJECT>/<SeqN>/imageSequences/video_camN.avi.

    seq = None nếu đường dẫn không có thư mục SeqN (khi đó cần --x-subject/--x-camera).
    """
    path = path.expanduser().resolve()
    parts = path.parts
    seqs = [i for i, p in enumerate(parts) if re.fullmatch(r"Seq\d+", p)]
    seq = parts[seqs[-1]] if seqs else None
    subject = subject_override
    if subject is None and seqs and seqs[-1] > 0:
        subject = parts[seqs[-1] - 1]
    match = re.search(r"video_cam(\d+)", path.stem, re.IGNORECASE)
    camera = camera_override if camera_override is not None else (int(match.group(1)) if match else None)
    if not subject or camera is None:
        raise SystemExit(
            f"Không suy ra được subject/camera từ video '{path}'. "
            "Dùng --x-subject/--x-camera hoặc --y-subject/--y-camera để chỉ rõ."
        )
    if int(camera) < 0:
        raise SystemExit(f"Camera phải >= 0: {camera}")
    return {"subject": str(subject), "seq": seq, "camera": int(camera), "video": path}


def annot_path(src: dict, pattern: str) -> Path:
    """annot.mat của nguồn theo --annot-pattern. Mẫu mặc định chỉ có annot của Seq1, nên
    video SeqN khác (vd. Seq2) mà mẫu không chứa {seq} thì dừng, thay vì đọc nhầm annot Seq1."""
    seq = src.get("seq")
    if "{seq}" in pattern and seq is None:
        raise SystemExit(f"--annot-pattern có {{seq}} nhưng đường dẫn {src['video']} không có "
                         "thư mục SeqN.")
    if "{seq}" not in pattern and seq not in (None, "Seq1"):
        raise SystemExit(f"Video {src['video']} thuộc {seq}, nhưng --annot-pattern '{pattern}' "
                         "chỉ trỏ tới annot của Seq1. Thêm {seq} vào --annot-pattern.")
    return (REPO / pattern.format(**src)).resolve()


def open_source(src: dict, args) -> dict:
    """Mở video và annotation tương ứng; src chứa video, subject, seq, camera, start, end."""
    src["annot"] = annot_path(src, args.annot_pattern)
    if not src["video"].is_file():
        raise SystemExit(f"Không thấy video: {src['video']}")
    if not src["annot"].is_file():
        raise SystemExit(f"Không thấy annotation: {src['annot']}")
    src["j2d"], a3 = load_annot(src["annot"], src["camera"])
    src["body_px"] = depth_scale(a3, fit_intrinsics(src["j2d"], a3), args.ref_height)
    cap = cv2.VideoCapture(str(src["video"]))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video != len(src["j2d"]):           # MPI: video và annot cùng số frame (S8 cam0: 6468)
        print(f"  ⚠ {src['video'].name}: video {n_video} frame, annot {len(src['j2d'])} frame "
              f"({src['annot']}). Kiểm tra video và annot có cùng subject/seq/camera không.")
    src["n"] = min(n_video, len(src["j2d"]))
    src["size"] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    src["fps"] = cap.get(cv2.CAP_PROP_FPS)
    if src["end"] > src["n"]:
        cap.release()
        raise SystemExit(
            f"Đoạn {src['start']}:{src['end']} vượt số frame khả dụng của "
            f"{src['video']} (video/annot chỉ có {src['n']} frame)."
        )
    for _ in range(src["start"]):
        cap.grab()
    src["cap"] = cap
    return src


def person_scale(j2d: np.ndarray) -> float:
    """Cỡ người trên ảnh: chiều lớn hơn của khung xương 2D (đứng = cao, nằm = dài)."""
    return float(max(np.ptp(j2d[:, 0]), np.ptp(j2d[:, 1]), 40.0))


def depth_scale(a3: np.ndarray, K: dict, height: float = REF_HEIGHT_MM) -> np.ndarray:
    """Cỡ người trên ảnh theo độ sâu, từng frame: chiều cao (px) của người cao `height` mm
    đứng ở độ sâu pelvis, f * height / Z. Không đổi khi đổi tư thế (khác person_scale).

    a3: (F, 28, 3) mm. NaN ở frame không có độ sâu hợp lệ (person_alpha khi đó dùng
    person_scale).
    """
    z = np.asarray(a3, float)[:, PELVIS, 2]
    ok = np.isfinite(z) & (z > 100.0)
    out = np.full(len(z), np.nan)
    out[ok] = np.sqrt(K["fx"] * K["fy"]) * height / z[ok]
    return out


def shrink(pts: np.ndarray, s: float, center) -> np.ndarray:
    """Thu nhỏ s quanh center (≈ cùng thân người lùi ra độ sâu Z/s nếu center = điểm chính)."""
    c = np.asarray(center, float)
    return c + s * (np.asarray(pts, float) - c)

# ============================================================
# 2. MẶT NẠ NGƯỜI (chroma key ∩ vùng khung xương)
# ============================================================


def body_region(j2d: np.ndarray, shape, scale: float, width: float) -> np.ndarray:
    """Vùng cho phép quanh khung xương: xương dày, thân đặc, đầu tròn."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(j2d).astype(np.int32)
    th = max(3, int(width * scale))
    for a, b in BONES:
        cv2.line(m, tuple(pts[a]), tuple(pts[b]), 255, th)
    cv2.fillPoly(m, [pts[TORSO]], 255)
    cv2.circle(m, tuple(pts[HEAD]), max(3, int(0.08 * scale)), 255, -1)
    return m > 0


def person_alpha(img: np.ndarray, j2d: np.ndarray, tau: int, width: float,
                 scale: Optional[float] = None, chroma: float = MIN_CHROMA):
    """-> (alpha float32 trong ROI, mặt nạ bool trong ROI, (x0, y0, x1, y1)) hoặc None.

    scale: cỡ người (px) quyết định độ dày vùng xương; nên lấy từ depth_scale. Bỏ trống
    hoặc NaN thì dùng person_scale(j2d) (khung 2D, hụt chân khi ngồi xổm).
    chroma: chỉ coi là phông khi thêm (G - max(R,B)) / G > chroma, tức xanh đậm màu chứ không
    chỉ ngả xanh. 0 = cách cũ: quần áo xám-xanh, da bị phông hắt xanh bị coi là phông.
    """
    h, w = img.shape[:2]
    s = float(scale) if scale is not None and np.isfinite(scale) else person_scale(j2d)
    pad = int(0.15 * s) + 10
    x0, y0 = np.floor(j2d.min(0)).astype(int) - pad
    x1, y1 = np.ceil(j2d.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    local = j2d - (x0, y0)
    region = body_region(local, (y1 - y0, x1 - x0), s, width)
    roi = img[y0:y1, x0:x1].astype(np.int16)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    excess = g - np.maximum(b, r)
    # nền xanh: trội xanh rõ, hoặc bóng tối nhưng vẫn ngả xanh (giày đen R≈G≈B không bị loại)
    green = (excess > tau) | ((excess > 3) & (g > 0.42 * (b + g + r + 1)))
    if chroma > 0:
        green &= excess > chroma * np.maximum(g, 1)
    fg = (~green & region).astype(np.uint8)
    k = max(3, int(0.02 * s) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fg = fg > 0
    holes, n_holes = label(binary_fill_holes(fg) & ~fg)
    if n_holes:                                  # chỉ lấp lỗ nhỏ (phản xạ xanh trên áo),
        area = np.bincount(holes.ravel())        # không lấp khe giữa hai chân / tay
        small = np.flatnonzero(area < 0.002 * s * s)
        fg |= np.isin(holes, small[small > 0])
    core = body_region(local, fg.shape, s, 0.0)          # xương mảnh: chỉ giữ mảng chạm khung xương
    lab, _ = label(fg)
    keep = np.unique(lab[core & fg])
    fg = np.isin(lab, keep[keep > 0])
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (5, 5), 0)
    return alpha, fg, (x0, y0, x1, y1)


def mask_fill(fg: np.ndarray, j2d: np.ndarray, box, width: float,
              scale: Optional[float] = None) -> float:
    """Tỉ lệ vùng quanh khung xương (độ dày width) được giữ làm người.

    S8: trung vị ~0.6. Thấp bất thường (< FILL_WARN) nghĩa là chroma key đang xoá quần áo
    hoặc da; khi đó người đứng sau lộ ra đè lên người đứng trước ở chỗ bị xoá.
    """
    x0, y0 = box[:2]
    s = float(scale) if scale is not None and np.isfinite(scale) else person_scale(j2d)
    region = body_region(j2d - (x0, y0), fg.shape, s, width)
    return float(fg[region].sum() / max(int(region.sum()), 1))


def paste_scaled(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, box,
                 s: float = 1.0, center=(0.0, 0.0)) -> np.ndarray:
    """Trộn src[box] (thu nhỏ s quanh center) vào dst theo alpha; cắt phần ra ngoài ảnh.

    Trả về mặt nạ bool cỡ dst của phần đã dán (alpha > 0.5).
    """
    x0, y0, x1, y1 = box
    H, W = dst.shape[:2]
    mask = np.zeros((H, W), bool)
    nx0, ny0 = np.round(shrink((x0, y0), s, center)).astype(int)
    w, h = max(1, int(round((x1 - x0) * s))), max(1, int(round((y1 - y0) * s)))
    roi, al = src[y0:y1, x0:x1], alpha
    if s != 1.0:
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        al = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    a0, b0, a1, b1 = max(nx0, 0), max(ny0, 0), min(nx0 + w, W), min(ny0 + h, H)
    if a1 <= a0 or b1 <= b0:
        return mask
    part = roi[b0 - ny0:b1 - ny0, a0 - nx0:a1 - nx0].astype(np.float32)
    al = al[b0 - ny0:b1 - ny0, a0 - nx0:a1 - nx0]
    out = dst[b0:b1, a0:a1].astype(np.float32)
    dst[b0:b1, a0:a1] = (al[..., None] * part + (1 - al[..., None]) * out).round().astype(np.uint8)
    mask[b0:b1, a0:a1] = al > 0.5
    return mask

# ============================================================
# 3. GHÉP VÀ XUẤT
# ============================================================


def ffmpeg_writer(path: Path, size, fps: float):
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def unused_edits(edit_dir: Path, n: int) -> list:
    """Tệp trong frames_edit/ mà render sẽ KHÔNG dùng: không phải <số 6 chữ số>.png
    (vd. .jpg, tên khác) hoặc số frame >= n."""
    names = {f"{i:06d}.png" for i in range(n)}
    return sorted(p.name for p in edit_dir.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.name not in names)


def render_frames(out: Path, meta: dict) -> list:
    """Ghép video từ frames_raw/, ưu tiên frames_edit/ nếu frame đã được sửa.

    Trả về danh sách số frame (0-based, frame của video ghép) lấy từ frames_edit/.
    """
    raw_dir = out / "frames_raw"
    edit_dir = out / "frames_edit"
    edit_dir.mkdir(parents=True, exist_ok=True)
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    w0, h0 = map(int, meta["size"])
    scale = float(meta.get("out_scale", 1.0))
    size = (int(round(w0 * scale)) // 2 * 2, int(round(h0 * scale)) // 2 * 2)

    ignored = unused_edits(edit_dir, n)
    if ignored:
        print(f"  ⚠ Bỏ qua {len(ignored)} tệp trong frames_edit/ (chỉ dùng PNG tên 000000.png.."
              f"{n - 1:06d}.png): {', '.join(ignored[:8])}")
    selected, missing, edited = [], [], []
    for i in range(n):
        name = f"{i:06d}.png"
        fixed, original = edit_dir / name, raw_dir / name
        if fixed.is_file():
            selected.append(fixed)
            edited.append(i)
        elif original.is_file():
            selected.append(original)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} frame. Ví dụ: {', '.join(missing[:8])}")
    edited_count = len(edited)

    writer = ffmpeg_writer(out / "composite.mp4", size, fps)
    try:
        for i, path in enumerate(selected):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Không đọc được frame: {path}")
            if img.shape[1::-1] != size:
                raise SystemExit(
                    f"Kích thước {path.name}: {img.shape[1]}x{img.shape[0]}, "
                    f"cần {size[0]}x{size[1]}. Hãy giữ nguyên kích thước khi sửa bằng Paint.")
            writer.stdin.write(img.tobytes())
            if i % 100 == 0:
                print(f"\r  Render: {i}/{n}", end="", flush=True)
    finally:
        writer.stdin.close()
        code = writer.wait()
    if code != 0:
        raise SystemExit(f"ffmpeg kết thúc với mã lỗi {code}")
    print(f"\r  Đã render {n} frame; dùng {edited_count} frame đã chỉnh sửa.")
    return edited


def mark_rendered(out: Path, edited: list):
    """Ghi vào overlay_meta.json: video = composite.mp4, frame nào đã sửa tay.

    overlay_frames.csv (vis_y, diện tích) và các file GT KHÔNG đổi theo frame sửa: chỉ sửa
    lỗi mặt nạ, không xoá / thêm người.
    """
    path = out / "overlay_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update({"video": "composite.mp4", "edited_frames": edited,
                 "rendered": datetime.now().isoformat(timespec="seconds")})
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def draw_preview(img, frame: int, vis_y: float, jx, jy, path: Path):
    canvas = img.copy()
    for j, color, name in ((jx, (40, 40, 230), "X"), (jy, (230, 150, 0), "Y")):
        x0, y0 = j.min(0).astype(int)
        x1, y1 = j.max(0).astype(int)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 5)
        cv2.putText(canvas, name, (x0, max(y0 - 12, 30)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
    cv2.putText(canvas, f"f{frame} visY={vis_y:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 255, 255), 5)
    k = 768 / max(img.shape[:2])
    cv2.imwrite(str(path), cv2.resize(canvas, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))


def composite(X: dict, Y: dict, n: int, out: Path, args) -> pd.DataFrame:
    size = X["size"]
    center = (size[0] / 2.0, size[1] / 2.0)
    out_size = (int(round(size[0] * args.out_scale)) // 2 * 2,
                int(round(size[1] * args.out_scale)) // 2 * 2)
    # all: chỉ MP4; generate: chỉ PNG (~3 MB/frame ở 2048², ~4.5 GB / 1500 frame) để sửa tay
    keep_png = args.mode == "generate" and not args.preview_only
    writer = None if args.mode == "generate" or args.preview_only else ffmpeg_writer(
        out / "composite.mp4", out_size, X["fps"])
    raw_dir = out / "frames_raw"
    if keep_png:
        raw_dir.mkdir(parents=True, exist_ok=True)
        edit_dir = out / "frames_edit"
        old = sorted(edit_dir.glob("*.png")) if edit_dir.is_dir() else []
        if old:
            print(f"  ⚠ frames_edit/ còn {len(old)} frame sửa từ lần trước; --mode render sẽ dùng "
                  "chúng thay frame mới. Xoá đi nếu không còn đúng.")
    if args.preview_every:
        (out / "preview").mkdir(exist_ok=True)
    rows = []
    try:
        for t in range(n):
            okx, img_x = X["cap"].read()
            oky, img_y = Y["cap"].read()
            if not (okx and oky):
                print(f"\n  Video hết ở frame {t}; dừng.")
                break
            sx, sy = X["start"] + t, Y["start"] + t
            jx, jy = X["j2d"][sx], Y["j2d"][sy]
            img = img_x.copy()
            px = person_alpha(img_x, jx, args.tau, args.width, X["body_px"][sx], args.min_chroma)
            py = person_alpha(img_y, jy, args.tau, args.width, Y["body_px"][sy], args.min_chroma)
            fill_x = (mask_fill(px[1], jx, px[2], args.width, X["body_px"][sx]) if px is not None
                      else np.nan)
            fill_y = (mask_fill(py[1], jy, py[2], args.width, Y["body_px"][sy]) if py is not None
                      else np.nan)
            my = (paste_scaled(img, img_y, py[0], py[2], args.y_scale, center) if py is not None
                  else np.zeros(img.shape[:2], bool))
            mx = (paste_scaled(img, img_x, px[0], px[2]) if px is not None     # X luôn đứng trước
                  else np.zeros(img.shape[:2], bool))
            area_x, area_y = np.count_nonzero(mx), np.count_nonzero(my)
            vis_y = 1 - np.count_nonzero(mx & my) / area_y if area_y else 1.0
            y_cut = bool(touches_border(jy, Y["size"]))
            rows.append((t, sx, sy, area_x, area_y, vis_y, y_cut, fill_x, fill_y))
            if args.preview_every and t % args.preview_every == 0:
                draw_preview(img, t, vis_y, jx, shrink(jy, args.y_scale, center),
                             out / "preview" / f"f{t:05d}.jpg")
            if args.out_scale != 1.0:            # PNG / MP4 ở đúng kích thước video cuối cùng
                img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
            if keep_png:
                cv2.imwrite(str(raw_dir / f"{t:06d}.png"), img)
            if writer is not None:
                writer.stdin.write(img.tobytes())
            if t % 100 == 0:
                print(f"\r  Ghép: {t}/{n}", end="", flush=True)
    finally:
        if writer is not None:
            writer.stdin.close()
            writer.wait()
    print(f"\r  Đã ghép {len(rows)} frame.        ")
    return pd.DataFrame(rows, columns=["frame", "src_x", "src_y", "area_x", "area_y", "vis_y",
                                       "y_cut", "fill_x", "fill_y"])


def rel_path(path, base: Path) -> str:
    """Đường dẫn tương đối so với thư mục kết quả (chép cả repo sang chỗ khác vẫn đọc được);
    khác ổ đĩa trên Windows thì giữ tuyệt đối. Bên đọc dùng (out / p).resolve() cho cả hai."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def write_meta(out: Path, X: dict, Y: dict, frames: pd.DataFrame, args):
    """X/Y["end"] là cuối đoạn THỰC SỰ dùng (start + số frame ghép), không phải END đã nhập."""
    center = [X["size"][0] / 2.0, X["size"][1] / 2.0]
    persons = {}
    for S, layer, col, s in ((X, "front", "src_x", 1.0), (Y, "back", "src_y", args.y_scale)):
        persons[f"{'X' if layer == 'front' else 'Y'}_{S['subject']}"] = {
            "layer": layer, "subject": S["subject"], "seq": S.get("seq"), "camera": S["camera"],
            "start": S["start"], "end": S["end"],
            "video": rel_path(S["video"], out), "annot": rel_path(S["annot"], out),
            "src_col": col, "scale": s, "center": center}
    meta = {
        "version": 2, "name": args.name, "created": datetime.now().isoformat(timespec="seconds"),
        "fps": X["fps"], "size": list(X["size"]), "out_scale": args.out_scale,
        "n_frames": len(frames), "persons": persons,
        "chroma_tau": args.tau, "chroma_min_ratio": args.min_chroma,
        "body_width": args.width, "ref_height_mm": args.ref_height,
        "frames_csv": "overlay_frames.csv",
        "video": None if (args.preview_only or args.mode == "generate") else "composite.mp4",
        "frames_dir": "frames_raw" if (args.mode == "generate" and not args.preview_only) else None,
    }
    (out / "overlay_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    frames.to_csv(out / "overlay_frames.csv", index=False, float_format="%.4f")


def joint_cover(jx: np.ndarray, jy: np.ndarray, s: float, center, width: float) -> np.ndarray:
    """Ước lượng nhanh tỉ lệ Y bị X che, từng frame: phần khớp của Y (đã thu nhỏ) nằm trong
    "ống" bán kính width/2 * cỡ người quanh các xương của X. Thường cao hơn tỉ lệ che thật
    (đo bằng mặt nạ khi ghép) khoảng 1.5-2 lần."""
    y = shrink(jy, s, center)                                               # (n, 28, 2)
    a = jx[:, [p for p, _ in BONES]]
    ab = jx[:, [q for _, q in BONES]] - a                                    # (n, 27, 2)
    t = np.einsum("nkd,nmkd->nmk", ab, y[:, :, None] - a[:, None])
    t = np.clip(t / np.maximum((ab ** 2).sum(-1), 1e-6)[:, None], 0, 1)
    d = np.linalg.norm(y[:, :, None] - (a[:, None] + t[..., None] * ab[:, None]), axis=-1).min(-1)
    r = width / 2 * np.array([person_scale(k) for k in jx])
    return (d < r[:, None]).mean(1)


def touches_border(j2d: np.ndarray, size, margin: int = 10) -> np.ndarray:
    """Khung xương (…, 28, 2) chạm/vượt mép ảnh nguồn: người bị khung hình cắt mất một phần.

    Với Y (thu nhỏ, đặt giữa ảnh ghép) chỗ bị cắt thành một đường thẳng giữa cảnh.
    """
    w, h = size
    return ((j2d[..., 0].min(-1) < margin) | (j2d[..., 0].max(-1) > w - margin)
            | (j2d[..., 1].min(-1) < margin) | (j2d[..., 1].max(-1) > h - margin))


def intervals_overlap(a_start: int, b_start: int, n: int, min_gap: int = 0) -> bool:
    """Hai đoạn nguồn dài n frame [a, a+n), [b, b+n) giao nhau hoặc cách nhau < min_gap frame."""
    return not (a_start + n + min_gap <= b_start or b_start + n + min_gap <= a_start)


def same_timeline(X: dict, Y: dict) -> bool:
    """Cùng một lần quay (cùng annot.mat) -> cùng dòng thời gian, KỂ CẢ khác camera:
    S8 cam0 và S8 cam4 ở cùng frame là cùng một động tác nhìn từ hai phía."""
    return Path(X["annot"]).resolve() == Path(Y["annot"]).resolve()


def check_overlap(X: dict, Y: dict, n: int, min_gap: int, allow: bool = False):
    """Cùng dòng thời gian mà hai đoạn n frame giao nhau / cách nhau < min_gap: dừng (hoặc chỉ
    cảnh báo nếu allow). Phần giao là cùng một chuỗi động tác hai lần: Y là X thu nhỏ."""
    if not (same_timeline(X, Y) and intervals_overlap(X["start"], Y["start"], n, min_gap)):
        return
    msg = (f"X [{X['start']}, {X['start'] + n}) và Y [{Y['start']}, {Y['start'] + n}) cùng dòng "
           f"thời gian và giao nhau / cách nhau < {min_gap} frame: cùng động tác sẽ xuất hiện hai lần.")
    if not allow:
        raise SystemExit(msg + " Đổi đoạn (xem --suggest), giảm --frames, hoặc thêm --allow-overlap.")
    print(f"  ⚠ {msg} (--allow-overlap)")


def suggest(X: dict, Y: dict, n: int, args, step: int = 250, lo=0.10, hi=0.45):
    """In các cặp (start X, start Y) cho clip n frame có X che Y vừa phải và nhiều lần.

    Cùng dòng thời gian thì loại mọi cặp có đoạn nguồn giao nhau (cách nhau < --min-gap),
    vì phần giao là cùng một chuỗi động tác xuất hiện hai lần trong clip. Chưa loại động
    tác lặp lại ở hai thời điểm không giao nhau (cần so độ giống chuyển động, ví dụ DTW).
    """
    center = (X["size"][0] / 2.0, X["size"][1] / 2.0)
    same = same_timeline(X, Y)
    y_cut = touches_border(Y["j2d"][:Y["n"]], Y["size"])
    rows = []
    for sx in range(X["start"], X["end"] - n + 1, step):
        for sy in range(Y["start"], Y["end"] - n + 1, step):
            if same and intervals_overlap(sx, sy, n, args.min_gap):
                continue
            cut = y_cut[sy:sy + n].mean()
            if cut > args.max_y_cut:
                continue
            cov = joint_cover(X["j2d"][sx:sx + n], Y["j2d"][sy:sy + n], args.y_scale,
                              center, args.width)
            on = cov > 0.1
            episodes = int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))
            if lo <= on.mean() <= hi:
                rows.append((episodes, on.mean(), np.mean(cov > 0.5), cut, sx, sy))
    print(f"Clip {n} frame, Y x{args.y_scale:g}: các cặp có Y bị che >10% ở {lo:.0%}-{hi:.0%} số frame "
          "(ước lượng từ khung xương, thường cao hơn thực tế 1.5-2 lần); "
          f"Y bị mép khung nguồn cắt <= {args.max_y_cut:.0%} số frame"
          + (f"; đoạn nguồn không giao nhau, cách nhau >= {args.min_gap} frame" if same else ""))
    print("  start X  start Y   che>10%  che>50%  Y cắt mép  số lần bị che")
    for e, a, b, cut, sx, sy in sorted(rows, key=lambda r: (r[0], r[1]), reverse=True)[:10]:
        print(f"  {sx:7d}  {sy:7d}   {a:7.1%}  {b:7.1%}  {cut:9.1%}  {e:8d}")
    if not rows:
        print("  (không có cặp nào; thử đổi --frames hoặc --y-scale)")


def summarize(frames: pd.DataFrame):
    v = frames["vis_y"]
    print(f"  Y bị X che >10%: {(v < 0.9).mean():.1%} số frame | >50%: {(v < 0.5).mean():.1%} | "
          f"che hoàn toàn (>95%): {(v < 0.05).mean():.1%}")
    if "y_cut" in frames and frames["y_cut"].any():
        print(f"  ⚠ Y bị mép khung nguồn cắt ở {frames['y_cut'].mean():.1%} số frame "
              "(thành đường cắt thẳng giữa ảnh ghép; cột y_cut trong overlay_frames.csv)")
    for who, col in (("X", "fill_x"), ("Y", "fill_y")):
        if col not in frames:
            continue
        fill = frames[col].median()
        print(f"  Mặt nạ {who} giữ {fill:.0%} vùng quanh khung xương (trung vị; S8 ~60%)")
        if fill < FILL_WARN:
            print(f"  ⚠ Mặt nạ {who} hụt: chroma key có thể đang xoá quần áo/da màu xanh, người đứng "
                  "sau sẽ lộ ra đè lên chỗ bị xoá. Xem mask_qa.jpg; thử tăng --min-chroma.")


# ============================================================
# 4. GROUND TRUTH CỦA VIDEO GHÉP
# ============================================================


def load_annot(mat: Path, camera: int):
    """-> (annot2 (F, 28, 2) px, annot3 (F, 28, 3) mm, hệ camera)."""
    data = loadmat(mat, squeeze_me=True, struct_as_record=False)
    a2 = np.asarray(data["annot2"][camera], float)
    a3 = np.asarray(data["annot3"][camera], float)
    return a2.reshape(len(a2), -1, 2), a3.reshape(len(a3), -1, 3)


def fit_intrinsics(a2: np.ndarray, a3: np.ndarray) -> dict:
    """fx, fy, cx, cy khớp annot3 -> annot2 (bình phương tối thiểu trên mọi khớp, mọi frame).

    MPI chiếu annot3 ra annot2 không méo, nên sai số ~0 px; không cần file calib.
    """
    ok = np.isfinite(a2).all(-1) & np.isfinite(a3).all(-1) & (a3[..., 2] > 1)
    x, y = a3[ok][:, 0] / a3[ok][:, 2], a3[ok][:, 1] / a3[ok][:, 2]
    fx, cx = np.linalg.lstsq(np.c_[x, np.ones_like(x)], a2[ok][:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[y, np.ones_like(y)], a2[ok][:, 1], rcond=None)[0]
    err = np.hypot(fx * x + cx - a2[ok][:, 0], fy * y + cy - a2[ok][:, 1])
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "fit_err_px_median": float(np.median(err))}


def project(P: np.ndarray, K: dict) -> np.ndarray:
    return np.stack([K["fx"] * P[..., 0] / P[..., 2] + K["cx"],
                     K["fy"] * P[..., 1] / P[..., 2] + K["cy"]], -1)


def scene_pose(P: np.ndarray, uv_root: np.ndarray, K_own: dict, K_scene: dict,
               s: float) -> np.ndarray:
    """Thân người (annot3 của camera gốc) đặt vào camera của video ghép.

    Tịnh tiến cứng cả thân sao cho pelvis có độ sâu Z' = Z * f_scene / (s * f_own) (cùng cỡ
    trên ảnh) và chiếu đúng vào uv_root (vị trí pelvis đã thu nhỏ trên ảnh ghép).
    f = sqrt(fx * fy): ảnh được thu nhỏ đều hai trục, nên nếu tỉ lệ fy/fx của hai camera khác
    nhau thì không độ sâu nào khớp đúng cả hai trục; trung bình nhân chia đều sai số.
    Pelvis khớp chính xác; các khớp khác sai số phối cảnh yếu (vài px).
    """
    root = P[:, PELVIS]
    f = lambda K: np.sqrt(K["fx"] * K["fy"])  # noqa: E731
    z = root[:, 2] * f(K_scene) / (s * f(K_own))
    target = np.stack([(uv_root[:, 0] - K_scene["cx"]) * z / K_scene["fx"],
                       (uv_root[:, 1] - K_scene["cy"]) * z / K_scene["fy"], z], -1)
    return P + (target - root)[:, None]


def gt_table(frames: np.ndarray, person: str, src: np.ndarray, pts: np.ndarray,
             axes: str) -> pd.DataFrame:
    df = pd.DataFrame({"frame": frames, "person": person, "src_frame": src})
    for name, j in GT_JOINTS.items():
        for k, a in enumerate(axes):
            df[f"{name}_{a}"] = pts[:, j, k]
    return df


def export_gt(out: Path) -> dict:
    """Ghi GT của video ghép vào out/ (đọc overlay_meta.json, overlay_frames.csv):

      gt_3d.csv        pose 3D gốc (annot3, mm, hệ camera của video nguồn từng người);
                       đây là GT dùng cho PA-MPJPE (bất biến dịch/xoay/tỉ lệ)
      gt_3d_scene.csv  pose 3D trong camera của video ghép (= camera X, mm): X giữ nguyên,
                       Y tịnh tiến ra sau (độ sâu ~Z/s) để chiếu đúng chỗ đã dán; cho C_root
      gt_2d.csv        khớp 2D trên ảnh ghép (px): X = annot2, Y = tâm + s * (annot2 - tâm)
    Cột: frame, person, src_frame (frame trong video nguồn), <khớp>_x/_y/_z hoặc _u/_v.
    """
    meta_path = out / "overlay_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = pd.read_csv(out / meta["frames_csv"])
    k = float(meta.get("out_scale", 1.0))
    persons = meta["persons"]
    front = next(n for n, p in persons.items() if p.get("layer") == "front")
    loaded = {n: load_annot((out / p["annot"]).resolve(), int(p["camera"])) for n, p in persons.items()}
    K = {n: fit_intrinsics(*loaded[n]) for n in persons}
    K_scene = K[front]
    t = frames["frame"].to_numpy()
    t3, ts, t2, check = [], [], [], {}
    for name, p in persons.items():
        a2, a3 = loaded[name]
        src = frames[p["src_col"]].to_numpy()
        s, c = float(p["scale"]), np.asarray(p["center"], float)
        P, uv = a3[src], c + s * (a2[src] - c)
        Ps = P if name == front else scene_pose(P, uv[:, PELVIS], K[name], K_scene, s)
        err = np.linalg.norm(project(Ps, K_scene) - uv, axis=-1)[:, list(GT_JOINTS.values())]
        size = np.array([person_scale(q) for q in uv])                    # cỡ người trên ảnh ghép
        check[name] = {
            "reproj_px_median": float(np.median(err)),
            "reproj_px_p95": float(np.percentile(err, 95)),
            "reproj_rel_p95": float(np.percentile(err / size[:, None], 95)),
            "pelvis_reproj_px_max": float(err[:, 0].max()),
            "per_joint_px": {j: {"median": float(np.median(err[:, i])),
                                 "p95": float(np.percentile(err[:, i], 95))}
                             for i, j in enumerate(GT_JOINTS)},
            "k_diff_vs_scene": {a: float(K[name][a] - K_scene[a]) for a in ("fx", "fy", "cx", "cy")},
            "aspect_diff": float(K[name]["fy"] / K[name]["fx"] - K_scene["fy"] / K_scene["fx"]),
        }
        t3.append(gt_table(t, name, src, P, "xyz"))
        ts.append(gt_table(t, name, src, Ps, "xyz"))
        t2.append(gt_table(t, name, src, uv * k, "uv"))
    for fname, tabs in (("gt_3d.csv", t3), ("gt_3d_scene.csv", ts), ("gt_2d.csv", t2)):
        pd.concat(tabs).sort_values(["frame", "person"]).to_csv(out / fname, index=False,
                                                                float_format="%.3f")
    scene_K = {a: K_scene[a] * k for a in ("fx", "fy", "cx", "cy")}
    meta["gt"] = {"files": ["gt_3d.csv", "gt_3d_scene.csv", "gt_2d.csv"],
                  "units": {"3d": "mm", "2d": "px"}, "joints": list(GT_JOINTS),
                  "intrinsics_fit": K, "scene_camera": {"person": front, **scene_K},
                  "scene_reprojection": check}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta["gt"]


def mask_qa(out: Path, n: int = 8, min_sep: int = 50, per_row: int = 4) -> Path:
    """Ảnh kiểm tra mặt nạ ở các tư thế khó của từng người trong clip -> out/mask_qa.jpg.

    Độ khó = (rộng / cao của khung xương) + (khoảng cách cổ tay xa nhất tới cổ / cỡ người):
    cao khi nằm, cúi sâu, dang tay. Mỗi ô: ảnh gốc | phần được giữ (nền tím = bị bỏ).
    """
    meta = json.loads((out / "overlay_meta.json").read_text(encoding="utf-8"))
    frames = pd.read_csv(out / meta["frames_csv"])
    tau, width = int(meta.get("chroma_tau", 12)), float(meta.get("body_width", 0.16))
    chroma = float(meta.get("chroma_min_ratio", 0.0))   # clip cũ: chỉ có tau
    height = meta.get("ref_height_mm")         # clip cũ (không có khoá này): cỡ theo khung 2D
    rows = []
    for name, p in meta["persons"].items():
        j, a3 = load_annot((out / p["annot"]).resolve(), int(p["camera"]))
        src = frames[p["src_col"]].to_numpy()
        js = j[src]
        body = (depth_scale(a3, fit_intrinsics(j, a3), float(height))[src] if height
                else np.full(len(src), np.nan))
        w, h = np.ptp(js[..., 0], 1), np.ptp(js[..., 1], 1)
        reach = np.linalg.norm(js[:, [11, 16]] - js[:, [NECK]], axis=-1).max(1) / np.maximum(w, h)
        pick = []
        for i in np.argsort(-(w / np.maximum(h, 1) + reach)):
            if all(abs(i - q) > min_sep for q in pick):
                pick.append(int(i))
            if len(pick) == n:
                break
        cap = cv2.VideoCapture(str((out / p["video"]).resolve()))
        tiles = []
        for i in sorted(pick):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src[i]))
            ok, img = cap.read()
            r = person_alpha(img, js[i], tau, width, body[i], chroma) if ok else None
            if r is None:
                continue
            alpha, _, (x0, y0, x1, y1) = r
            roi = img[y0:y1, x0:x1]
            cut = (roi * alpha[..., None] + np.array([255, 0, 255]) * (1 - alpha[..., None]))
            pair = np.hstack([roi, cut.astype(np.uint8)])
            pair = cv2.resize(pair, (int(pair.shape[1] * 260 / pair.shape[0]), 260))
            cv2.putText(pair, f"{name} t{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(pair)
        for i in range(0, len(tiles), per_row):              # mỗi hàng per_row cặp ảnh
            rows.append(np.hstack([np.pad(t, ((0, 0), (0, 6), (0, 0)), constant_values=255)
                                   for t in tiles[i:i + per_row]]))
    width_max = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 6), (0, width_max - r.shape[1]), (0, 0)), constant_values=255)
                       for r in rows])
    path = out / "mask_qa.jpg"
    k = min(1.0, 2400 / sheet.shape[1])
    cv2.imwrite(str(path), cv2.resize(sheet, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    return path


def print_gt(out: Path, gt: dict):
    print(f"  GT -> {out}: {', '.join(gt['files'])}")
    for name, K in gt["intrinsics_fit"].items():
        print(f"    {name}: fx={K['fx']:.1f} fy={K['fy']:.1f} cx={K['cx']:.1f} cy={K['cy']:.1f} "
              f"(khớp annot3->annot2: {K['fit_err_px_median']:.2f} px)")
    for name, c in gt["scene_reprojection"].items():
        kd = c["k_diff_vs_scene"]
        if name != gt["scene_camera"]["person"]:
            print(f"    {name} - camera cảnh: Δfx={kd['fx']:+.2f} Δfy={kd['fy']:+.2f} "
                  f"Δcx={kd['cx']:+.2f} Δcy={kd['cy']:+.2f} px, Δ(fy/fx)={c['aspect_diff']:+.5f}")
            if max(abs(v) for v in kd.values()) > 2.0 or abs(c["aspect_diff"]) > 1e-3:
                print("      ⚠ Intrinsics hai nguồn khác nhau đáng kể: kiểm tra trước khi tạo hàng loạt.")
        worst = sorted(c["per_joint_px"].items(), key=lambda kv: -kv[1]["p95"])[:3]
        print(f"    {name}: chiếu gt_3d_scene so với gt_2d: trung vị {c['reproj_px_median']:.1f} px, "
              f"p95 {c['reproj_px_p95']:.1f} px ({c['reproj_rel_p95']:.1%} cỡ người), "
              f"pelvis tối đa {c['pelvis_reproj_px_max']:.2f} px; p95 lớn nhất: "
              + ", ".join(f"{j} {v['p95']:.1f}" for j, v in worst))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Ghép hai đoạn video: X đứng trước, Y thu nhỏ và đứng sau. "
            "--x/--y là khoảng frame START:END (END không được lấy)."
        ),
        epilog=(
            "Ví dụ (V=../WHAM/inputs/mpi_inf_3dhpe/S8/Seq1/imageSequences/video_cam0.avi):\n"
            "  %(prog)s --x 0:6468 --y 0:6468 --x-video $V --frames 1500 --suggest\n"
            "  %(prog)s --x 0:1500 --y 4750:6250 --x-video $V\n"
            "  %(prog)s --x 0:1500 --y 4750:6250 --x-video $V --mode generate\n"
            "  %(prog)s --mode render --name ovl_S8c0f0-1500_S8c0f4750-6250_s0.5\n"
            "Ghi chú: subject/camera được suy ra từ đường dẫn dạng <SUBJECT>/Seq1/imageSequences/video_cam<CAMERA>.avi. "
            "Nếu đường dẫn không theo dạng này, dùng --x-subject/--x-camera và --y-subject/--y-camera. "
            "Bỏ --y-video thì Y dùng cùng video với X; khi đó hai đoạn không được giao nhau "
            "(cách nhau >= --min-gap frame), trừ khi có --allow-overlap. --frames là số frame đầu ra tối đa. "
            "--y-scale=0.5 làm chiều rộng và chiều cao của Y còn một nửa (diện tích còn 1/4)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--x", required=False, metavar="START:END", help="khoảng frame lấy làm người X (END không được lấy)")
    p.add_argument("--y", required=False, metavar="START:END", help="khoảng frame lấy làm người Y (END không được lấy)")
    p.add_argument("--x-video", type=Path, metavar="PATH", help="đường dẫn video gốc; bắt buộc khi tạo video ghép")
    p.add_argument("--y-video", type=Path, metavar="PATH", help="video nguồn của Y; mặc định dùng cùng video với X")
    p.add_argument("--x-subject", help="subject dùng để đọc annot.mat của X (mặc định suy ra từ đường dẫn video)")
    p.add_argument("--x-camera", type=int, help="camera dùng để đọc annot.mat của X (mặc định suy ra từ tên video_camN)")
    p.add_argument("--y-subject", help="subject dùng để đọc annot.mat của Y (mặc định suy ra từ đường dẫn video)")
    p.add_argument("--y-camera", type=int, help="camera dùng để đọc annot.mat của Y (mặc định suy ra từ tên video_camN)")
    p.add_argument("--mode", choices=("all", "generate", "render"), default="all",
                   help="all: ghép thẳng ra MP4 (không lưu PNG); generate: chỉ lưu từng frame PNG vào "
                        "frames_raw/ để sửa tay (~3 MB/frame ở 2048²); render: ghép MP4 từ frames_raw/, "
                        "ưu tiên frames_edit/ (cần --name)")
    p.add_argument("--gt-only", nargs="+", type=Path, metavar="DIR", help="chỉ (ghi lại) file GT cho video ghép đã có")
    p.add_argument("--mask-qa", nargs="+", type=Path, metavar="DIR", help="chỉ vẽ mask_qa.jpg cho video ghép đã có")
    p.add_argument("--y-scale", type=float, default=0.5, metavar="SCALE",
                   help="hệ số thu nhỏ đồng đều chiều rộng/cao của Y (mặc định: 0.5; diện tích còn 1/4)")
    p.add_argument("--frames", type=int, help="số frame đầu ra tối đa (mặc định: độ dài ngắn hơn của hai đoạn)")
    p.add_argument("--name", help="tên thư mục kết quả (mặc định tự sinh)")
    p.add_argument("--out-root", type=Path, default=REPO / "outputs" / "overlay")
    p.add_argument("--tau", type=int, default=12, help="ngưỡng chroma key: G - max(R,B) > tau là nền")
    p.add_argument("--min-chroma", type=float, default=MIN_CHROMA,
                   help="chroma key: chỉ là nền khi thêm (G - max(R,B)) / G > giá trị này (mặc định "
                        f"{MIN_CHROMA}); 0 = cách cũ, áo xám-xanh hay da hắt xanh bị coi là nền")
    p.add_argument("--width", type=float, default=0.16, help="độ dày vùng xương / cỡ người")
    p.add_argument("--ref-height", type=float, default=REF_HEIGHT_MM, help="chiều cao tham chiếu (mm) cho cỡ người theo độ sâu")
    p.add_argument("--out-scale", type=float, default=1.0, help="thu nhỏ cả video xuất")
    p.add_argument("--preview-every", type=int, default=250, help="lưu frame mẫu mỗi N frame (0 = tắt)")
    p.add_argument("--preview-only", action="store_true", help="chỉ lưu frame mẫu, không xuất video")
    p.add_argument("--suggest", action="store_true", help="chỉ in các cặp vị trí bắt đầu X/Y có mức che phù hợp")
    p.add_argument("--min-gap", type=int, default=100,
                   help="cùng dòng thời gian (cùng annot.mat): hai đoạn nguồn phải cách nhau ít nhất N frame "
                        "(khi ghép và khi --suggest)")
    p.add_argument("--allow-overlap", action="store_true",
                   help="cho phép hai đoạn cùng dòng thời gian giao nhau (không khuyến khích: Y lặp lại động "
                        "tác của X)")
    p.add_argument("--max-y-cut", type=float, default=0.0, help="--suggest: tỉ lệ frame tối đa Y bị mép khung nguồn cắt")
    p.add_argument("--annot-pattern", default=ANNOT_PATTERN)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gt_only or args.mask_qa:
        for out in args.gt_only or []:
            print_gt(out, export_gt(out))
        for out in args.mask_qa or []:
            print(f"  Kiểm tra mặt nạ -> {mask_qa(out)}")
        return

    if args.mode == "render":
        if not args.name:
            raise SystemExit("--mode render cần --name để xác định thư mục kết quả.")
        out = (args.out_root / args.name).resolve()
        meta_path = out / "overlay_meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"Không thấy metadata: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mark_rendered(out, render_frames(out, meta))
        print(f"  Video: {out / 'composite.mp4'}")
        return

    if not (args.x and args.y):
        raise SystemExit("Cần --x và --y (hoặc --gt-only DIR / --mask-qa DIR). Dùng --help để xem cách chạy.")
    if args.frames is not None and args.frames <= 0:
        raise SystemExit("--frames phải lớn hơn 0.")
    if args.y_scale <= 0:
        raise SystemExit("--y-scale phải lớn hơn 0.")
    if args.out_scale <= 0:
        raise SystemExit("--out-scale phải lớn hơn 0.")
    if args.x_video is None:
        raise SystemExit("Cần --x-video để chỉ rõ video gốc.")
    x_seg, y_seg = parse_segment(args.x), parse_segment(args.y)
    X = open_source({**video_identity(args.x_video, args.x_subject, args.x_camera), **x_seg}, args)
    y_video = args.y_video if args.y_video is not None else args.x_video
    y_subject = args.y_subject if args.y_subject is not None else (args.x_subject if args.y_video is None else None)
    y_camera = args.y_camera if args.y_camera is not None else (args.x_camera if args.y_video is None else None)
    Y = open_source({**video_identity(y_video, y_subject, y_camera), **y_seg}, args)
    if abs(X["fps"] - Y["fps"]) > 0.01:
        raise SystemExit(f"fps khác nhau: X {X['fps']} / Y {Y['fps']}")
    if X["size"] != Y["size"]:
        raise SystemExit(f"Kích thước khác nhau: X {X['size']} / Y {Y['size']}")
    if args.suggest:
        n_suggest = min(args.frames or 1500, X["end"] - X["start"], Y["end"] - Y["start"])
        suggest(X, Y, n_suggest, args)

        print("\nBỏ --suggest để ra được video overlay.")
        return

    n = min(X["end"] - X["start"], Y["end"] - Y["start"])
    n = min(n, args.frames) if args.frames is not None else n
    if n <= 0:
        raise SystemExit("Không còn frame chung sau khi lệch START.")
    check_overlap(X, Y, n, args.min_gap, args.allow_overlap)
    for S in (X, Y):                         # đoạn thực sự dùng (tên thư mục, meta)
        S["end"] = S["start"] + n

    args.name = args.name or (f"ovl_{X['subject']}c{X['camera']}f{X['start']}-{X['end']}"
                              f"_{Y['subject']}c{Y['camera']}f{Y['start']}-{Y['end']}_s{args.y_scale:g}")
    out = (args.out_root / args.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"X = {X['video']} | {X['subject']} cam{X['camera']} [{X['start']}:{X['end']}) (trước)")
    print(f"Y = {Y['video']} | {Y['subject']} cam{Y['camera']} [{Y['start']}:{Y['end']}) "
          f"(sau, x{args.y_scale:g}) | {n} frame, {X['fps']} fps")

    frames = composite(X, Y, n, out, args)
    for S in (X, Y):                         # video hết sớm: chỉ ghi phần đã ghép
        S["end"] = S["start"] + len(frames)
    write_meta(out, X, Y, frames, args)
    print_gt(out, export_gt(out))
    summarize(frames)
    if args.mode == "generate":
        print(f"  Đã sinh frame PNG: {out / 'frames_raw'}")
        print(f"  Sau khi sửa, chạy --mode render --name {args.name}")
    else:
        print(f"  Kết quả: {out}")


if __name__ == "__main__":
    main()
