#!/usr/bin/env python3
"""
Gán body_id của WHAM với người thật trong Ground Truth -> mapping.txt.

Quy trình cho mỗi sequence trong tập được chọn (validation / test) của manifest:
  1. Đã có mapping.txt -> bỏ qua.
  2. pkl WHAM -> RA-WHAM.csv -> video animation WHAM (wham_anim.mp4).
  3. Ground Truth (annot.mat của MPI-INF-3DHP, poses3d/*.npy của Harmony4D — thư mục,
     tệp poses3d.zip hoặc Drive File ID của zip đó —, hoặc
     overlay_meta.json của video ghép 2 người từ scripts/make_overlay_pair.py)
     -> video animation GT (gt_anim.mp4).
     Video ghép có thể gán nhãn tự động bằng scripts/overlay_autolabel.py.
  4. Có output_wham_video (video WHAM đã render sẵn trên video gốc)
     -> ghép 3 video theo thứ tự:
        wham_anim.mp4 | output_wham_video | gt_anim.mp4
     Không có -> ghép 2 video:
        wham_anim.mp4 | gt_anim.mp4
  5. Mở video ghép, hỏi từng body_id thuộc người GT nào, ghi mapping.txt.
     body_id không được gán = không phải người GT.

Các video animation đánh số theo frame của video gốc (frame không có detection
vẫn được giữ), nên 2-3 video luôn chạy khớp nhau.

Chạy:
  python3 scripts/wham_gt_mapping.py                     # hỏi tập rồi chạy
  python3 scripts/wham_gt_mapping.py --split validation
  python3 scripts/wham_gt_mapping.py --list              # xem trạng thái
  python3 scripts/wham_gt_mapping.py --split test --only h4d_041_grappling2_cam07 --force
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "configs" / "splits.json"
SPLITS = ("validation", "test")
VIDEO_H = 720

# ============================================================
# 1. KHUNG XƯƠNG
# ============================================================

WHAM_SMPL_IDX = {  # chỉ số khớp SMPL-24 (theo wham_all_subjects_to_csv.py)
    "right_shoulder": 17, "right_elbow": 19, "right_wrist": 21,
    "left_shoulder": 16, "left_elbow": 18, "left_wrist": 20,
    "right_hip": 2, "right_knee": 5, "right_ankle": 8,
    "left_hip": 1, "left_knee": 4, "left_ankle": 7,
}
LIMBS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
LIMB_BONES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
              ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
              ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
              ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
              ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
              ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
# 13 khớp MPI-INF-3DHP (thứ tự 28 khớp, zero-based; theo annot_mat_to_csv.py)
MPI_IDX = {"neck": 5, "left_shoulder": 9, "right_shoulder": 14, "left_elbow": 10,
           "right_elbow": 15, "left_wrist": 11, "right_wrist": 16, "left_hip": 18,
           "right_hip": 23, "left_knee": 19, "right_knee": 24, "left_ankle": 20,
           "right_ankle": 25}
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"] + LIMBS

# 12 khớp chung: thứ tự bắt buộc cho WHAM-2D và GT-2D.
WHAM_2D_JOINTS = LIMBS
OVERLAY_CSV_JOINTS_12 = LIMBS

# annot2 và annot3 của MPI cùng dùng thứ tự 28 khớp.
MPI_ANNOT2_TO_SHARED_12 = [
    MPI_IDX[j] for j in LIMBS
]
# [9, 14, 10, 15, 11, 16, 18, 23, 19, 24, 20, 25]

# Harmony4D poses2d: SMPL 45-joint layout.
H4D_45_TO_SHARED_12 = [
    WHAM_SMPL_IDX[j] for j in LIMBS
]
# [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

def make_skeleton(joints, extra_bones=()):
    """Khung xương = tên khớp + cặp chỉ số xương."""
    pos = {j: i for i, j in enumerate(joints)}
    bones = [(pos[a], pos[b]) for a, b in list(LIMB_BONES) + list(extra_bones)]
    top = [pos["left_shoulder"], pos["right_shoulder"]]
    bottom = [pos["left_ankle"], pos["right_ankle"]]
    return {"joints": joints, "bones": bones, "top": top, "bottom": bottom}


OVERLAY_CSV_JOINTS = ["pelvis", "neck"] + LIMBS

SKELETONS = {
    "wham": make_skeleton(["pelvis"] + LIMBS),

    "mpi13": make_skeleton(
        list(MPI_IDX),
        [
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
        ],
    ),

    "coco17": make_skeleton(
        COCO17,
        [
            ("nose", "left_eye"),
            ("nose", "right_eye"),
            ("left_eye", "left_ear"),
            ("right_eye", "right_ear"),
            ("nose", "left_shoulder"),
            ("nose", "right_shoulder"),
        ],
    ),

    "mpi_overlay_csv": make_skeleton(
        OVERLAY_CSV_JOINTS,
        [
            ("pelvis", "neck"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
        ],
    ),
}
PALETTE = [(40, 40, 230), (230, 150, 0), (40, 180, 40), (200, 40, 200), (0, 200, 230),
           (0, 120, 255), (160, 160, 0), (120, 40, 160), (80, 80, 80), (40, 200, 160)]

# ============================================================
# 2. MANIFEST
# ============================================================


def is_google_drive_file_id(value) -> bool:
    """Nhận diện Drive File ID (chuỗi ID, không phải URL hoặc đường dẫn)."""
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{20,}", value) is not None


# Kiểu GT được phép ghi gt.path là Drive File ID (harmony4d: ID của poses3d.zip).
DRIVE_GT_TYPES = ("mpi_overlay_csv", "overlay_csv", "harmony4d")
H4D_GT_MARKER = ".gt_source"   # ghi nguồn của bản GT đã giải nén, để biết khi nào giải nén lại


def load_manifest(path: Path) -> dict:
    """Đọc manifest; wham_pkl, gt.path (GT CSV, poses3d.zip của harmony4d) và
    output_wham_video có thể là Drive File ID hoặc đường dẫn cục bộ."""
    m = json.loads(path.read_text(encoding="utf-8"))
    base = (path.parent / m.get("base_dir", ".")).resolve()

    m["smpl_model"] = (base / m["smpl_model"]).resolve()
    m["output_dir"] = (base / m["output_dir"]).resolve()

    for split in SPLITS:
        for seq in m.get(split, []):
            source = seq["wham_pkl"]
            seq["wham_pkl_is_drive"] = is_google_drive_file_id(source)

            if not seq["wham_pkl_is_drive"]:
                seq["wham_pkl"] = (base / source).resolve()

            # Ground Truth có thể là đường dẫn local hoặc Google Drive File ID.
            gt_type = seq["gt"]["type"]
            gt_source = seq["gt"]["path"]

            seq["gt"]["path_is_drive"] = is_google_drive_file_id(gt_source)

            if seq["gt"]["path_is_drive"]:
                if gt_type not in DRIVE_GT_TYPES:
                    raise ValueError(
                        f"Sequence '{seq['name']}': Google Drive File ID "
                        f"chỉ được hỗ trợ cho GT dạng CSV hoặc harmony4d "
                        f"(poses3d.zip), không phải '{gt_type}'."
                    )

                # Giữ nguyên File ID trong manifest.
                # Đường dẫn file tải về sẽ được xác định khi xử lý sequence.
            else:
                seq["gt"]["path"] = (base / gt_source).resolve()

            # Tên mới: output_wham_video.
            # Tên cũ overlay_video chỉ được giữ để tương thích manifest cũ.
            new_video = seq.get("output_wham_video")
            old_video = seq.get("overlay_video")

            if new_video and old_video:
                new_path = (base / new_video).resolve()
                old_path = (base / old_video).resolve()

                if new_path != old_path:
                    raise ValueError(
                        f"Sequence '{seq['name']}' khai báo cả "
                        "output_wham_video và overlay_video với hai "
                        "đường dẫn khác nhau. Chỉ giữ output_wham_video."
                    )

            if new_video:
                video = new_video
            elif old_video:
                print(
                    f"Cảnh báo: sequence '{seq['name']}' vẫn dùng "
                    "'overlay_video'. Hãy đổi thành 'output_wham_video'."
                )
                video = old_video
            else:
                video = ""

            seq["out_dir"] = m["output_dir"] / seq["name"]

            # output_wham_video cũng có thể là Drive File ID: giữ ID, còn đường dẫn
            # là bản cache trong out_dir (tên theo ID). Chỉ tải khi cần ghép video
            # (fetch_output_wham_video trong build_videos); --list không tải.
            seq["output_wham_video_drive"] = (
                video if is_google_drive_file_id(video) else None
            )
            if seq["output_wham_video_drive"]:
                seq["output_wham_video"] = seq["out_dir"] / f"output_wham_{video}.mp4"
            else:
                seq["output_wham_video"] = (
                    (base / video).resolve() if video else None
                )

            # Không sử dụng tên cũ trong các bước xử lý phía sau.
            seq.pop("overlay_video", None)

    return m


def sequence_status(seq: dict) -> str:
    if seq.get("skip") is True:
        return "SKIP (skip=true)"

    if (seq["out_dir"] / "mapping.txt").is_file():
        return "đã có mapping"

    gt = seq["gt"]

    if gt.get("path_is_drive"):
        # File ID được xem là nguồn GT sẵn sàng tải.
        pass
    elif not gt["path"].exists():
        return "THIẾU GT"

    if (seq["out_dir"] / "RA-WHAM.csv").is_file():
        return "đã có CSV, sẵn sàng"

    if seq.get("wham_pkl_is_drive"):
        return "PKL Google Drive, sẵn sàng tải"

    if not seq["wham_pkl"].exists():
        return "THIẾU pkl"

    return "chưa gán"


def print_status(manifest: dict):
    all_seqs = [
        seq
        for split in SPLITS
        for seq in manifest.get(split, [])
    ]
    name_width = max(
        28,
        max((len(seq["name"]) for seq in all_seqs), default=0),
    )

    for split in SPLITS:
        print(f"\n[{split}]")

        for seq in manifest.get(split, []):
            video = seq.get("output_wham_video")
            has_video = video is not None and video.is_file()

            n_videos = "3 video" if has_video else "2 video"

            if seq.get("output_wham_video_drive") and not has_video:
                n_videos = "3 video (Drive)"       # tải khi ghép video
            elif video is not None and not has_video:
                n_videos += " (thiếu output)"

            print(
                f"  {seq['name']:<{name_width}} "
                f"{n_videos:<22} "
                f"{sequence_status(seq)}"
            )


def ask_split() -> str:
    print("\nGán nhãn cho tập nào?")
    for i, s in enumerate(SPLITS, 1):
        print(f"  {i}. {s}")
    while True:
        ans = input("Chọn (1/2): ").strip().lower()
        if ans in ("1", "2"):
            return SPLITS[int(ans) - 1]
        if ans in SPLITS:
            return ans
        print("Chỉ nhập 1 hoặc 2.")

# ============================================================
# 3. WHAM: PKL -> CSV -> POSE THEO FRAME
# ============================================================


class _ChumpyStub:
    """SMPL_NEUTRAL.pkl có object chumpy; ta chỉ cần J_regressor nên bỏ qua chúng."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["state"] = state


class _SmplUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def load_j_regressor(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        model = _SmplUnpickler(f, encoding="latin1").load()
    J = model["J_regressor"]
    J = J.toarray() if scipy.sparse.issparse(J) else np.asarray(J)
    if J.shape != (24, 6890):
        raise ValueError(f"J_regressor sai shape {J.shape}: {path}")
    return J


def wham_subject_table(sid, d, J) -> pd.DataFrame:
    """Một track WHAM -> bảng RA-WHAM (như wham_all_subjects_to_csv.py, ground=sequence)."""
    verts = d["verts"].astype(np.float64)
    jc = np.einsum("jv,nvc->njc", J, verts)
    R = (Rotation.from_rotvec(d["pose_world"][:, :3])
         * Rotation.from_rotvec(d["pose"][:, :3]).inv()).as_matrix()
    pelvis_cam = jc[:, 0]
    pelvis_w = d["trans_world"] + (pelvis_cam - d["trans"])
    joints = np.einsum("nij,npj->npi", R, jc - pelvis_cam[:, None]) + pelvis_w[:, None]
    vw = np.einsum("nij,npj->npi", R, verts - pelvis_cam[:, None]) + pelvis_w[:, None]
    joints[:, :, 1] -= np.percentile(vw[:, :, 1].min(axis=1), 10)
    root = joints[:, 0].copy()
    rel = joints - root[:, None]
    df = pd.DataFrame({"frame_id": np.asarray(d["frame_ids"]).astype(int),
                       "body_id": int(sid), "pelvis_x": -root[:, 0],
                       "pelvis_y": root[:, 2], "pelvis_z": root[:, 1]})
    for name, idx in WHAM_SMPL_IDX.items():
        df[f"{name}_x"], df[f"{name}_y"], df[f"{name}_z"] = (
            -rel[:, idx, 0], rel[:, idx, 2], rel[:, idx, 1])
    return df


def download_drive_file(file_id: str, destination: Path, what: str = "PKL") -> Path:
    """Tải một tệp (mặc định PKL) từ Google Drive công khai bằng File ID (cần cài gdown)."""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Thiếu thư viện gdown. Cài bằng: pip install gdown") from exc
    print(f"  Tải {what} từ Google Drive (ID: {file_id}) ...")
    downloaded = gdown.download(id=file_id, output=str(destination), quiet=False)
    if not downloaded or not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Không tải được {what}. Kiểm tra File ID và quyền chia sẻ 'Anyone with the link'."
        )
    print(f"  Đã tải {destination.stat().st_size:,} bytes.")
    return destination


def fetch_output_wham_video(seq: dict) -> Path | None:
    """Đường dẫn output_wham_video trên máy; nếu manifest ghi Drive ID thì tải một lần.

    Video tải về được giữ trong out_dir làm cache (tên theo ID, nên đổi ID thì tải
    bản mới). Tải lỗi không chặn việc gán nhãn: trả về đường dẫn chưa có tệp và
    build_videos sẽ cảnh báo rồi ghép 2 video.
    """
    video = seq.get("output_wham_video")
    file_id = seq.get("output_wham_video_drive")
    if not file_id or video is None or (video.is_file() and video.stat().st_size > 0):
        return video
    video.parent.mkdir(parents=True, exist_ok=True)
    print(f"  output_wham_video là Google Drive File ID -> tải về {video.name}")
    try:
        download_drive_file(file_id, video, what="video WHAM")
    except Exception as exc:                 # mạng, quyền chia sẻ, thiếu gdown
        print(f"  CẢNH BÁO: không tải được output_wham_video: {exc}")
    finally:                                 # không để sót tệp tải dở của gdown
        for part in video.parent.glob(video.name + "*.part"):
            part.unlink(missing_ok=True)
    return video


def h4d_npy_dir(root: Path) -> Path | None:
    """Thư mục chứa *.npy trong bản giải nén: ưu tiên thư mục tên poses3d, rồi thư
    mục nông nhất (zip có thể có sẵn poses3d/ hoặc để .npy ngay ở gốc)."""
    dirs = {p.parent for p in root.rglob("*.npy")}
    if not dirs:
        return None
    return min(dirs, key=lambda d: (d.name != "poses3d", len(d.parts), str(d)))


def cached_h4d_gt(cache_dir: Path, source: str) -> Path | None:
    """Thư mục *.npy đã giải nén trước đó, nếu nó được giải nén từ đúng nguồn này."""
    marker = cache_dir / H4D_GT_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8") == source:
        return h4d_npy_dir(cache_dir)
    return None


def extract_h4d_gt_zip(zip_path: Path, cache_dir: Path, source: str) -> Path:
    """Giải nén zip poses3d vào cache_dir rồi trả về thư mục chứa *.npy.

    Giải nén vào thư mục tạm, kiểm tra có .npy, ghi dấu nguồn rồi mới đổi tên:
    bị ngắt hay zip hỏng thì không để lại bản giải nén dở bị dùng nhầm lần sau.
    """
    tmp = cache_dir.with_name(cache_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
                zf.extractall(tmp, members)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"GT không phải tệp zip hợp lệ: {zip_path}") from exc
        if h4d_npy_dir(tmp) is None:
            raise ValueError(f"Không có tệp .npy nào trong {zip_path.name}")
        (tmp / H4D_GT_MARKER).write_text(source, encoding="utf-8")
        shutil.rmtree(cache_dir, ignore_errors=True)
        tmp.rename(cache_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return h4d_npy_dir(cache_dir)


def resolve_h4d_gt_zip(seq: dict, gt: dict) -> Path:
    """GT harmony4d dạng poses3d.zip (Drive File ID hoặc tệp .zip trên máy): giải nén
    một lần vào out_dir (gt_<ID>/ hoặc gt_<tên zip>/), trả về thư mục chứa *.npy.

    Dùng lại bản đã giải nén nếu cùng nguồn (cùng ID; zip trên máy: cùng đường dẫn,
    kích thước, thời điểm sửa). Zip tải từ Drive bị xoá sau khi giải nén.
    """
    out_dir, is_drive = seq["out_dir"], gt.get("path_is_drive", False)
    if is_drive:
        file_id = str(gt["path"])
        zip_path, cache_dir = out_dir / f"gt_{file_id}.zip", out_dir / f"gt_{file_id}"
        source = f"drive:{file_id}"
    else:
        zip_path = Path(gt["path"])
        if not zip_path.is_file():
            raise FileNotFoundError(f"Không thấy tệp GT: {zip_path}")
        st = zip_path.stat()
        cache_dir = out_dir / f"gt_{zip_path.stem}"
        source = f"local:{zip_path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    cached = cached_h4d_gt(cache_dir, source)
    if cached is not None:
        print(f"  Dùng lại GT đã giải nén: {cache_dir.name}/")
        return cached
    out_dir.mkdir(parents=True, exist_ok=True)
    if not is_drive:
        print(f"  Giải nén GT {zip_path.name} -> {cache_dir.name}/")
        return extract_h4d_gt_zip(zip_path, cache_dir, source)
    try:
        download_drive_file(file_id, zip_path, what="GT poses3d.zip")
        print(f"  Giải nén GT -> {cache_dir.name}/")
        return extract_h4d_gt_zip(zip_path, cache_dir, source)
    finally:                                 # bỏ zip đã tải và tệp tải dở của gdown
        zip_path.unlink(missing_ok=True)
        for part in out_dir.glob(zip_path.name + "*.part"):
            part.unlink(missing_ok=True)


def resolve_gt_path(seq: dict) -> dict:
    """
    Chuẩn bị Ground Truth để xử lý.

    - GT local: giữ nguyên đường dẫn (harmony4d: thư mục poses3d).
    - harmony4d là tệp .zip hoặc Drive File ID của poses3d.zip: giải nén một lần
      vào thư mục output của sequence (resolve_h4d_gt_zip).
    - GT CSV trên Google Drive: tải về thư mục output của sequence.
    - Cache được định danh bằng File ID, tránh dùng nhầm file khi manifest đổi ID.

    Trả về bản sao cấu hình GT, trong đó path luôn là Path local.
    Không thay đổi File ID gốc trong seq.
    """
    gt = dict(seq["gt"])

    if gt["type"] == "harmony4d" and (
        gt.get("path_is_drive", False) or str(gt["path"]).lower().endswith(".zip")
    ):
        gt["path"] = resolve_h4d_gt_zip(seq, gt)
        return gt

    if not gt.get("path_is_drive", False):
        return gt

    if gt["type"] not in DRIVE_GT_TYPES:
        raise ValueError(
            f"Google Drive GT hiện chỉ hỗ trợ {', '.join(DRIVE_GT_TYPES)}, "
            f"không phải '{gt['type']}'."
        )

    file_id = str(gt["path"])

    if not is_google_drive_file_id(file_id):
        raise ValueError(f"Google Drive File ID không hợp lệ: {file_id}")

    cache_path = seq["out_dir"] / f"gt_{file_id}.csv"

    if cache_path.is_file() and cache_path.stat().st_size > 0:
        print(f"  Dùng lại GT CSV: {cache_path.name}")
    else:
        print(f"  Tải Ground Truth từ Google Drive: {file_id}")
        download_drive_file(file_id, cache_path)

    gt["path"] = cache_path
    return gt

def wham_pkl_to_csv(pkl: Path, smpl: Path, out_csv: Path) -> pd.DataFrame:
    if out_csv.is_file():
        print(f"  Dùng lại CSV: {out_csv.name}")
        return pd.read_csv(out_csv)
    print(f"  Đọc pkl WHAM: {pkl}")
    results = joblib.load(pkl)
    J = load_j_regressor(smpl)
    tables = [wham_subject_table(sid, d, J) for sid, d in results.items()
              if len(d.get("frame_ids", []))]
    if not tables:
        raise ValueError(f"PKL không chứa track WHAM hợp lệ: {pkl}")
    df = pd.concat(tables, ignore_index=True).sort_values(["frame_id", "body_id"])
    df.to_csv(out_csv, index=False)
    print(f"  Đã lưu {len(df)} dòng, {df['body_id'].nunique()} body_id -> {out_csv.name}")
    return df


def load_wham_csv(seq: dict, smpl: Path) -> pd.DataFrame:
    """Đọc CSV có sẵn hoặc tải Drive PKL -> CSV; chỉ xóa bản tải sau convert OK."""
    out_csv = seq["out_dir"] / "RA-WHAM.csv"
    if out_csv.is_file():
        return wham_pkl_to_csv(Path(), smpl, out_csv)
    if not seq.get("wham_pkl_is_drive"):
        return wham_pkl_to_csv(seq["wham_pkl"], smpl, out_csv)

    with tempfile.NamedTemporaryFile(prefix="wham_drive_", suffix=".pkl", delete=False) as f:
        temp_pkl = Path(f.name)
    try:
        download_drive_file(seq["wham_pkl"], temp_pkl)
        df = wham_pkl_to_csv(temp_pkl, smpl, out_csv)
    except Exception:
        print(f"  Lỗi khi tải/chuyển đổi; giữ file tạm để kiểm tra: {temp_pkl}")
        raise
    else:
        temp_pkl.unlink(missing_ok=True)
        print("  Đã xóa PKL tạm trên máy (file gốc trên Drive không bị ảnh hưởng).")
        return df


def wham_pkl_to_2d(pkl: Path, smpl: Path, image_size) -> pd.DataFrame:
    """12 khớp SMPL của mọi detection WHAM chiếu lên ảnh: frame, body_id, <khớp>_u/_v (px).

    Dùng đúng camera mà WHAM giả định: f = hypot(W, H), điểm chính ở tâm ảnh. `verts`
    trong pkl nằm sẵn trong hệ camera đó (khác hệ world của RA-WHAM.csv), nên chiếu thẳng.
    Cột frame lấy từ frame_ids, cùng quy ước với cột frame_id của RA-WHAM.csv.
    """
    W, H = (float(x) for x in image_size)
    f = float(np.hypot(W, H))
    J = load_j_regressor(smpl)
    idx = [WHAM_SMPL_IDX[j] for j in LIMBS]
    tables = []
    for sid, d in joblib.load(pkl).items():
        frames = np.asarray(d.get("frame_ids", []), int).reshape(-1)
        if not len(frames):
            continue
        verts = np.asarray(d["verts"], np.float64)
        if len(verts) != len(frames):
            raise ValueError(f"body_id {sid}: {len(frames)} frame_ids nhưng {len(verts)} verts")
        jc = np.einsum("jv,nvc->njc", J, verts)[:, idx]
        z = jc[..., 2]
        ok = np.isfinite(jc).all(-1) & (z > 1e-6)
        z = np.where(ok, z, 1.0)
        u = np.where(ok, f * jc[..., 0] / z + W / 2, np.nan)
        v = np.where(ok, f * jc[..., 1] / z + H / 2, np.nan)
        df = pd.DataFrame({"frame": frames, "body_id": int(sid)})
        for i, name in enumerate(LIMBS):
            df[f"{name}_u"], df[f"{name}_v"] = u[:, i], v[:, i]
        tables.append(df)
    if not tables:
        raise ValueError(f"PKL không có detection nào: {pkl}")
    return pd.concat(tables, ignore_index=True).sort_values(["frame", "body_id"],
                                                            ignore_index=True)


def load_wham_2d(seq: dict, smpl: Path) -> pd.DataFrame:
    """Đọc out_dir/WHAM-2D.csv; chưa có thì tạo từ pkl (pkl Drive được tải lại một lần).

    RA-WHAM.csv không có toạ độ ảnh, còn pkl tải từ Drive đã bị xoá sau khi chuyển
    đổi, nên lần đầu mỗi sequence phải tải lại pkl. Cần khoá "image_size": [W, H]
    trong manifest = kích thước video mà WHAM đã chạy.
    """
    out = seq["out_dir"] / "WHAM-2D.csv"
    if out.is_file():
        return pd.read_csv(out)
    size = seq.get("image_size")
    if not size or len(size) != 2:
        raise ValueError(f"Sequence '{seq['name']}' thiếu \"image_size\": [W, H] trong manifest "
                         "(kích thước video WHAM đã chạy), cần để chiếu WHAM lên ảnh.")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not seq.get("wham_pkl_is_drive"):
        df = wham_pkl_to_2d(seq["wham_pkl"], smpl, size)
    else:
        with tempfile.NamedTemporaryFile(prefix="wham_drive_", suffix=".pkl", delete=False) as f:
            temp_pkl = Path(f.name)
        try:
            download_drive_file(seq["wham_pkl"], temp_pkl)
            df = wham_pkl_to_2d(temp_pkl, smpl, size)
        finally:                       # kể cả khi lỗi hay Ctrl+C: không để sót pkl / .part
            temp_pkl.unlink(missing_ok=True)
            for part in temp_pkl.parent.glob(temp_pkl.name + "*.part"):
                part.unlink(missing_ok=True)
    df.to_csv(out, index=False)
    print(f"  Đã lưu {len(df)} dòng toạ độ ảnh -> {out.name}")
    return df


def wham_poses(df: pd.DataFrame) -> dict:
    """{frame: {"id k": (13,3)}}, toạ độ world = pelvis + khớp tương đối."""
    poses: dict = {}
    for r in df.itertuples(index=False):
        root = np.array([r.pelvis_x, r.pelvis_y, r.pelvis_z])
        pts = [root] + [root + np.array([getattr(r, f"{j}_{a}") for a in "xyz"])
                        for j in LIMBS]
        poses.setdefault(int(r.frame_id), {})[f"id {int(r.body_id)}"] = np.array(pts)
    return poses


def body_id_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("body_id")["frame_id"]
    return pd.DataFrame({"first": g.min(), "last": g.max(), "frames": g.count()})

# ============================================================
# 4. GROUND TRUTH
# ============================================================


def mpi_annot3(mat: Path, camera: int) -> np.ndarray:
    """annot.mat: annot3[camera] (F, 28*3) mm, hệ toạ độ camera -> (F, 13, 3) mét."""
    data = loadmat(mat, squeeze_me=True, struct_as_record=False)
    ann = np.asarray(data["annot3"][camera], dtype=float)
    return ann.reshape(len(ann), -1, 3)[:, list(MPI_IDX.values())] / 1000.0


def mpi_gt_poses(mat: Path, camera: int, person: str) -> dict:
    ann = mpi_annot3(mat, camera)
    return {f: {person: ann[f]} for f in range(len(ann)) if np.isfinite(ann[f]).all()}


def overlay_gt_poses(meta_path: Path) -> dict:
    """Video ghép của scripts/make_overlay_pair.py: 2 người MPI, frame nguồn theo overlay_frames.csv.

    Thu nhỏ người trong ảnh không đổi pose 3D (PA-MPJPE bất biến tỉ lệ), nên dùng annot3 gốc.
    CHỈ dùng cho phép đo hình dạng pose (PA-MPJPE). Mỗi người ở hệ camera của video nguồn
    riêng, nên KHÔNG dùng để tính vị trí gốc hay khoảng cách giữa hai người trong cảnh:
    việc đó dùng gt_3d_scene.csv (scripts/make_overlay_pair.py, export_gt).
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base = meta_path.parent
    frames = pd.read_csv(base / meta["frames_csv"])
    poses: dict = {}
    for name, p in meta["persons"].items():
        ann = mpi_annot3((base / p["annot"]).resolve(), int(p["camera"]))
        src = frames[p["src_col"]].to_numpy()
        for f, s in zip(frames["frame"].to_numpy(), src):
            if np.isfinite(ann[s]).all():
                poses.setdefault(int(f), {})[name] = ann[s]
    return poses


def h4d_gt_poses(poses_dir: Path) -> dict:
    """poses3d/00001.npy ... : {tên người: (17, >=3)}; file thứ i = frame i."""
    files = sorted(poses_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"Không có .npy trong {poses_dir}")
    poses = {}
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True).item()
        poses[i] = {str(k): np.asarray(v, float)[:17, :3] for k, v in d.items()
                    if np.asarray(v).shape[0] >= 17}
    return poses


def load_gt(gt: dict) -> tuple[dict, str]:
    if gt["type"] == "mpi":
        return (
            mpi_gt_poses(
                gt["path"],
                int(gt["camera"]),
                str(gt["person"]),
            ),
            "mpi13",
        )

    if gt["type"] == "harmony4d":
        return (
            h4d_gt_poses(gt["path"]),
            "coco17",
        )

    if gt["type"] == "mpi_overlay":
        return (
            overlay_gt_poses(gt["path"]),
            "mpi13",
        )

    if gt["type"] in ("mpi_overlay_csv", "overlay_csv"):
        return (
            overlay_csv_gt_poses(gt["path"]),
            "mpi_overlay_csv",
        )

    raise ValueError(
        f"gt.type không hỗ trợ: {gt['type']}"
    )


def gt_names(poses: dict) -> list[str]:
    return sorted({p for people in poses.values() for p in people})

def overlay_csv_gt_poses(csv_path: Path) -> dict:
    """
    Đọc Ground Truth 3D từ gt_3d_scene.csv của MPI overlay.

    Cấu trúc:
        frame, person, src_frame,
        pelvis_x, pelvis_y, pelvis_z,
        neck_x, neck_y, neck_z, ...

    Mỗi dòng tương ứng với một người tại một frame composite.

    Trả về:
        {
            frame: {
                person: np.ndarray(shape=(14, 3))
            }
        }

    Lưu ý:
        - Dùng cột frame để đồng bộ với video composite.
        - src_frame chỉ là metadata của video nguồn.
        - Giữ nguyên tọa độ scene, không trừ pelvis.
        - Trả về đơn vị m.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy GT CSV: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )

    df.columns = df.columns.str.strip()

    # Kiểm tra các cột bắt buộc.
    required = {"frame", "person", "src_frame"}

    for joint in OVERLAY_CSV_JOINTS:
        for axis in "xyz":
            required.add(f"{joint}_{axis}")

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "GT CSV thiếu các cột bắt buộc:\n"
            + "\n".join(missing)
        )

    if df.empty:
        raise ValueError(f"GT CSV rỗng: {csv_path}")

    # Chuẩn hóa frame.
    df["frame"] = pd.to_numeric(
        df["frame"], errors="raise"
    ).astype(int)

    if df["frame"].lt(0).any():
        raise ValueError("GT CSV có frame âm.")

    # Chuẩn hóa tên người.
    if df["person"].isna().any():
        raise ValueError("GT CSV có dòng không có person.")

    df["person"] = df["person"].astype(str).str.strip()

    if df["person"].eq("").any():
        raise ValueError("GT CSV có person rỗng.")

    # Một người chỉ có một pose trong một frame.
    duplicated = df.duplicated(
        subset=["frame", "person"],
        keep=False,
    )

    if duplicated.any():
        examples = df.loc[
            duplicated, ["frame", "person"]
        ].head(10)

        raise ValueError(
            "GT CSV có nhiều dòng trùng frame/person:\n"
            f"{examples.to_string(index=False)}"
        )

    # Chuyển các cột tọa độ sang số.
    coord_cols = [
        f"{joint}_{axis}"
        for joint in OVERLAY_CSV_JOINTS
        for axis in "xyz"
    ]

    for col in coord_cols:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    poses = {}

    for row in df.itertuples(index=False):
        frame = int(row.frame)
        person = row.person

        pts = np.array(
            [
                [
                    getattr(row, f"{joint}_{axis}")
                    for axis in "xyz"
                ]
                for joint in OVERLAY_CSV_JOINTS
            ],
            dtype=np.float64,
        )

        # Bỏ qua pose không có tọa độ hợp lệ.
        if not np.isfinite(pts).any():
            continue

        # CSV lưu mm; chuẩn hóa về mét như mpi_gt_poses và h4d_gt_poses.
        poses.setdefault(frame, {})[person] = pts / 1000.0

    if not poses:
        raise ValueError(
            f"Không tìm thấy pose hợp lệ trong {csv_path}"
        )

    people = sorted(
        {
            person
            for frame_poses in poses.values()
            for person in frame_poses
        }
    )

    print(
        f"  Đọc GT CSV: {len(df)} dòng, "
        f"{len(poses)} frame, "
        f"{len(people)} người"
    )
    print(f"  Người: {', '.join(people)}")
    print(
        f"  Frame: {min(poses)} - {max(poses)}"
    )

    return poses
# ============================================================
# 5. VẼ ANIMATION (OpenCV, nhanh hơn matplotlib nhiều lần)
# ============================================================


def up_rotation(poses: dict, skel: dict) -> np.ndarray:
    """
    Ma trận xoay đưa hướng vai - mắt cá trung bình về trục +z.

    Bỏ qua các pose không đủ dữ liệu hợp lệ để xác định hướng đứng.
    """
    vecs = []

    for people in poses.values():
        for p in people.values():
            top_pts = p[skel["top"]]
            bottom_pts = p[skel["bottom"]]

            top_valid = top_pts[
                np.isfinite(top_pts).all(axis=1)
            ]
            bottom_valid = bottom_pts[
                np.isfinite(bottom_pts).all(axis=1)
            ]

            if len(top_valid) == 0 or len(bottom_valid) == 0:
                continue

            top = top_valid.mean(axis=0)
            bottom = bottom_valid.mean(axis=0)

            vec = top - bottom
            norm = np.linalg.norm(vec)

            if norm > 1e-8 and np.isfinite(norm):
                vecs.append(vec / norm)

    if not vecs:
        raise ValueError(
            "Không xác định được hướng đứng từ Ground Truth. "
            "Kiểm tra các khớp vai và mắt cá."
        )

    up = np.mean(vecs, axis=0)
    norm = np.linalg.norm(up)

    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError(
            "Hướng đứng trung bình không hợp lệ."
        )

    up /= norm

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(up @ z)

    if c < -0.999:
        return np.diag([1.0, -1.0, -1.0])

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    return np.eye(3) + vx + vx @ vx / (1 + c)


def view_2d(pts: np.ndarray, az=np.radians(-35), el=np.radians(20)) -> np.ndarray:
    """Chiếu trực giao (đã z-up) theo góc phương vị az và góc nhìn xuống el."""
    x = np.cos(az) * pts[..., 0] + np.sin(az) * pts[..., 1]
    d = -np.sin(az) * pts[..., 0] + np.cos(az) * pts[..., 1]
    v = pts[..., 2] * np.cos(el) + d * np.sin(el)
    return np.stack([x, v], -1)


def fit_view(poses: dict, R: np.ndarray, size: int) -> dict:
    allp = np.concatenate([p for people in poses.values() for p in people.values()])
    allp = allp[np.isfinite(allp).all(1)] @ R.T
    uv = view_2d(allp)
    lo, hi = np.percentile(uv, 1, 0), np.percentile(uv, 99, 0)
    span = max(hi - lo) * 1.15 + 0.3
    return {"R": R, "lo": (lo + hi) / 2 - span / 2, "s": (size - 60) / span, "size": size}


def to_px(pts: np.ndarray, view: dict) -> np.ndarray:
    uv = (view_2d(pts @ view["R"].T) - view["lo"]) * view["s"]
    return np.stack([uv[:, 0] + 30, view["size"] - 30 - uv[:, 1]], 1)


def draw_person(img, px: np.ndarray, skel: dict, color, label: str):
    ok = np.isfinite(px).all(1)
    for a, b in skel["bones"]:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)),
                     color, 3, cv2.LINE_AA)
    for p in px[ok]:
        cv2.circle(img, tuple(p.astype(int)), 4, color, -1, cv2.LINE_AA)
    if ok.any():
        head = px[ok][np.argmin(px[ok][:, 1])].astype(int)
        cv2.putText(img, label, (head[0] - 20, head[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_legend(img, names: list[str], visible: set, colors: dict):
    for i, n in enumerate(names):
        y = 60 + 26 * i
        c = colors[n] if n in visible else (190, 190, 190)
        cv2.rectangle(img, (VIDEO_H - 150, y - 14), (VIDEO_H - 136, y), c, -1)
        cv2.putText(img, n, (VIDEO_H - 128, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (40, 40, 40) if n in visible else (170, 170, 170), 1, cv2.LINE_AA)


def render_animation(poses: dict, skel_name: str, n_frames: int, fps: float,
                     out: Path, title: str):
    """Mỗi frame của video gốc = 1 frame animation (frame trống vẫn giữ)."""
    skel = SKELETONS[skel_name]
    view = fit_view(poses, up_rotation(poses, skel), VIDEO_H)
    names = sorted({p for ps in poses.values() for p in ps},
                   key=lambda s: (len(s), s))
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (VIDEO_H, VIDEO_H))
    for f in range(n_frames):
        img = np.full((VIDEO_H, VIDEO_H, 3), 248, np.uint8)
        people = poses.get(f, {})
        for name, pts in people.items():
            draw_person(img, to_px(pts, view), skel, colors[name], name)
        draw_legend(img, names, set(people), colors)
        cv2.putText(img, f"{title}   frame {f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (30, 30, 30), 2, cv2.LINE_AA)
        vw.write(img)
        if f % 500 == 0:
            print(f"\r  Vẽ {out.name}: {f}/{n_frames}", end="", flush=True)
    vw.release()
    tmp.replace(out)
    print(f"\r  Đã vẽ {out.name}: {n_frames} frame        ")

# ============================================================
# 6. VIDEO: FPS, GHÉP (CÓ SỐ FRAME), MỞ
# ============================================================


def video_info(path: Path) -> tuple[float, float]:
    """(fps, thời lượng giây) bằng ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    num, den = info["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den), float(info["format"]["duration"])


def panel_width(cap) -> int:
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return int(round(w * VIDEO_H / h / 2)) * 2


def stamp(img, text: str, org, scale=1.0):
    """Chữ trắng viền đen, đọc được trên mọi nền."""
    for color, th in (((0, 0, 0), 6), ((255, 255, 255), 2)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def banner(width: int, f: int, n: int) -> np.ndarray:
    bar = np.full((44, width, 3), 32, np.uint8)
    text = f"frame {f}  /  {n - 1}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(bar, text, ((width - tw) // 2, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def read_panels(caps, widths, last):
    """Đọc 1 frame mỗi video; video đã hết thì giữ frame cuối."""
    for i, cap in enumerate(caps):
        ok, img = cap.read()
        if ok:
            last[i] = cv2.resize(img, (widths[i], VIDEO_H), interpolation=cv2.INTER_AREA)
        elif last[i] is None:
            last[i] = np.zeros((VIDEO_H, widths[i], 3), np.uint8)
    return [p.copy() for p in last]


def combine_videos(
    videos: list[Path],
    out: Path,
    fps: float,
    n_frames: int,
    has_output_wham_video: bool,
):
    """
    Ghép ngang 2 hoặc 3 video theo đúng số frame.

    Khi có output_wham_video:
        WHAM animation | WHAM output video | GT animation

    Khi không có:
        WHAM animation | GT animation
    """
    caps = [cv2.VideoCapture(str(v)) for v in videos]

    widths = [panel_width(c) for c in caps]
    last = [None] * len(caps)
    total_w = sum(widths)

    tmp = out.with_suffix(".tmp.mp4")

    vw = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, VIDEO_H + 44),
    )

    print(f"  Ghép {len(videos)} video -> {out.name}")

    for f in range(n_frames):
        panels = read_panels(caps, widths, last)

        # Video output WHAM nằm ở panel thứ hai (index 1).
        if has_output_wham_video:
            stamp(panels[1], f"frame {f}", (16, 40))

        vw.write(
            np.vstack([
                banner(total_w, f, n_frames),
                np.hstack(panels),
            ])
        )

        if f % 200 == 0:
            print(
                f"\r  Ghép: {f}/{n_frames}",
                end="",
                flush=True,
            )

    vw.release()

    for cap in caps:
        cap.release()

    to_h264(tmp, out)

    print(
        f"\r  Đã ghép {out.name}: "
        f"{n_frames} frame        "
    )


def to_h264(src: Path, dst: Path):
    """Nén lại H.264 để QuickTime/trình duyệt mở được; xoá file tạm."""
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                   check=True)
    src.unlink()


def open_video(path: Path):
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        print(f"  Hãy mở video: {path}")

# ============================================================
# 7. CÚ PHÁP GÁN NHÃN THEO FRAME
# ============================================================
#   1                     cả track là người số 1
#   x  (hoặc Enter)       cả track không phải người GT
#   1, 2@121-125 140      mặc định người 1; frame 121-125 và 140 là người 2
#   [0-2]->2, [3-11]->1   liệt kê từng đoạn (dạng ->, = hoặc : đều được)

NOT_PERSON = ""
RANGE_RE = re.compile(r"^\[?\s*(\d+)\s*(?:-\s*(\d+))?\s*\]?$")
ARROW_RE = re.compile(r"^\[?\s*(\d+)\s*(?:-\s*(\d+))?\s*\]?\s*(?:->|=>|=|:)\s*(\S+)$")


def parse_person(tok: str, names: list[str]) -> str:
    tok = tok.strip()
    if tok.lower() in ("x", "-", "0", "none", ""):
        return NOT_PERSON
    if tok.isdigit() and 1 <= int(tok) <= len(names):
        return names[int(tok) - 1]
    if tok in names:
        return tok
    raise ValueError(f"không có người GT '{tok}'")


def parse_range(tok: str) -> tuple[int, int]:
    m = RANGE_RE.match(tok.strip())
    if not m:
        raise ValueError(f"đoạn frame không hợp lệ '{tok}'")
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    if b < a:
        raise ValueError(f"đoạn ngược '{tok}'")
    return a, b


def parse_item(item: str, names, rules, state):
    """Một mục giữa hai dấu phẩy: mặc định, P@đoạn..., [a-b]->P, hoặc đoạn nối tiếp @."""
    m = ARROW_RE.match(item)
    if m:
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        rules.append((a, b, parse_person(m.group(3), names)))
        state["at"] = None
    elif "@" in item:
        who, ranges = item.split("@", 1)
        state["at"] = parse_person(who, names)
        rules += [(*parse_range(t), state["at"]) for t in ranges.split()]
    elif state["at"] is not None and RANGE_RE.match(item.split()[0]):
        rules += [(*parse_range(t), state["at"]) for t in item.split()]
    elif state["default"] is None:
        state["default"] = parse_person(item, names)
    else:
        raise ValueError(f"có hai giá trị mặc định ('{item}')")


def parse_assignment(text: str, names: list[str]):
    """-> (người mặc định hoặc None, [(a, b, người), ...])."""
    rules, state = [], {"default": None, "at": None}
    items = [t.strip() for t in re.split(r"[,;]", text) if t.strip()]
    for item in items or [""]:
        parse_item(item, names, rules, state)
    return state["default"], rules


def to_ranges(frames) -> list[tuple[int, int]]:
    """[3,4,5,9,10] -> [(3,5), (9,10)]"""
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def fmt_ranges(ranges) -> str:
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def resolve_labels(track: np.ndarray, default, rules) -> dict:
    """{frame: người} cho mọi frame của track; báo lỗi nếu ngoài track/chồng/thiếu."""
    first, last, frames = int(track[0]), int(track[-1]), set(int(f) for f in track)
    labels: dict = {}
    for a, b, who in rules:
        if a < first or b > last:
            raise ValueError(f"đoạn {a}-{b} nằm ngoài track ({first}-{last})")
        for f in range(a, b + 1):
            if f in labels and labels[f] != who:
                raise ValueError(f"frame {f} bị gán hai lần")
            if f in frames:
                labels[f] = who
    missing = sorted(frames - set(labels))
    if missing and default is None:
        raise ValueError(f"chưa gán frame {fmt_ranges(to_ranges(missing))} "
                         "(thêm người mặc định, ví dụ '1, ...')")
    labels.update({f: default for f in missing})
    return dict(sorted(labels.items()))


def segments(labels: dict) -> list[tuple[int, int, str, int]]:
    """Gộp các frame liên tiếp (theo thứ tự track) cùng người: (đầu, cuối, người, số frame)."""
    segs = []
    for f, who in labels.items():
        if segs and segs[-1][2] == who:
            a, _, _, n = segs[-1]
            segs[-1] = (a, f, who, n + 1)
        else:
            segs.append((f, f, who, 1))
    return segs

# ============================================================
# 8. HỎI NGƯỜI DÙNG VÀ GHI MAPPING
# ============================================================


def print_mapping_help(summary: pd.DataFrame, names: list[str]):
    print("\n  body_id   frame đầu - cuối   số frame")
    for bid, r in summary.iterrows():
        print(f"  {bid:>7}   {r['first']:>6} - {r['last']:<6}   {r['frames']:>7}")
    people = "   ".join(f"{i}. {n}" for i, n in enumerate(names, 1))
    print(f"\n  Người trong Ground Truth:   {people}   x. không phải người")
    print("\n  Cách nhập:")
    print("    1                       cả track là người số 1")
    print("    x  (hoặc Enter)         cả track không phải người GT")
    print("    1, 2@121-125 140        mặc định người 1; frame 121-125 và 140 là người 2")
    print("    [0-2]->2, [3-11]->1     liệt kê từng đoạn frame")
    print("  Lệnh: v = mở lại video, b = quay lại, q = thoát không lưu")


def show_segments(labels: dict):
    print("    Kiểm tra:")
    for a, b, who, n in segments(labels):
        span = f"{a}" if a == b else f"{a}-{b}"
        print(f"      {span:>11}   {who or '(không phải người)':<20} ({n} frame)")


def ask_one(bid, track: np.ndarray, names: list[str]):
    """Hỏi 1 body_id. Trả về {frame: người} hoặc lệnh 'v'/'b'/'q'."""
    raw = input(f"  body_id {bid} [{track[0]}-{track[-1]}] -> ").strip()
    if raw.lower() in ("v", "b", "q"):
        return raw.lower()
    try:
        labels = resolve_labels(track, *parse_assignment(raw, names))
    except ValueError as e:
        print(f"    ✗ {e}. Nhập lại.")
        return None
    persons = set(labels.values())
    if len(persons) == 1:
        who = persons.pop()
        print(f"    ✓ {bid} = {who or '(không phải người GT)'} (cả track)")
        return labels
    show_segments(labels)
    ok = input("    Đúng chưa? [Y/n] -> ").strip().lower() in ("", "y", "yes", "c", "co")
    return labels if ok else None


def ask_mapping(tracks: dict, names: list[str], video: Path):
    """Hỏi lần lượt từng body_id. Trả về {body_id: {frame: người}}, None nếu thoát."""
    ids, mapping, i = list(tracks), {}, 0
    while i < len(ids):
        try:
            ans = ask_one(ids[i], tracks[ids[i]], names)
        except EOFError:
            return None
        if ans == "q":
            return None
        if ans == "v":
            open_video(video)
        elif ans == "b":
            i = max(0, i - 1)
        elif isinstance(ans, dict):
            mapping[ids[i]] = ans
            i += 1
    return mapping


def mapping_lines(bid, labels: dict) -> list[str]:
    persons = list(dict.fromkeys(labels.values()))
    if len(persons) == 1:
        who = persons[0]
        return [f"{bid} = {who}" if who else f"# {bid} = (không phải người GT)"]
    lines = []
    for who in persons:
        rng = fmt_ranges(to_ranges(f for f, w in labels.items() if w == who))
        lines.append(f"{bid} = {who} @ {rng}" if who
                     else f"# {bid} = (không phải người GT) @ {rng}")
    return lines


def write_mapping(out_dir: Path, mapping: dict, seq: dict):
    lines = ["# Ánh xạ WHAM body_id = người trong Ground Truth",
             f"# sequence: {seq['name']}",
             f"# wham_pkl: {seq['wham_pkl']}",
             f"# gt: {seq['gt']['path']}",
             "# 'id = người'          : cả track là người đó",
             "# 'id = người @ frames' : chỉ các frame liệt kê (track bị nhảy người)",
             "# frame không có dòng gán = không phải người GT"]
    for bid, labels in mapping.items():
        lines += mapping_lines(bid, labels)
    (out_dir / "mapping.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = [(f, bid, who) for bid, labels in mapping.items() for f, who in labels.items()]
    pd.DataFrame(rows, columns=["frame", "body_id", "person"]).sort_values(
        ["frame", "body_id"]).to_csv(out_dir / "mapping_frames.csv", index=False)
    print(f"  Đã lưu: {out_dir / 'mapping.txt'}")
    print(f"  Đã lưu: {out_dir / 'mapping_frames.csv'}")

# ============================================================
# 9. XỬ LÝ MỘT SEQUENCE VÀ CHƯƠNG TRÌNH CHÍNH
# ============================================================


def build_videos(
    seq: dict,
    wham_df: pd.DataFrame,
    gt: dict,
    gt_skel: str,
    rerender: bool,
) -> Path:
    out = seq["out_dir"]
    output_video = fetch_output_wham_video(seq)    # tải từ Drive nếu manifest ghi ID

    has_output_wham_video = (
        output_video is not None and output_video.is_file()
    )

    if output_video is not None and not has_output_wham_video:
        print(
            f"  CẢNH BÁO: Không tìm thấy output_wham_video:\n"
            f"  {output_video}\n"
            "  Sẽ chỉ ghép WHAM animation và GT animation."
        )

    # Nếu có video WHAM output thì lấy FPS từ video đó.
    fps = (
        video_info(output_video)[0]
        if has_output_wham_video
        else float(seq.get("fps", 25))
    )

    n = max(
        int(wham_df["frame_id"].max()) + 1,
        max(gt) + 1,
    )

    targets = [
        (
            out / "wham_anim.mp4",
            wham_poses(wham_df),
            "wham",
            "WHAM",
        ),
        (
            out / "gt_anim.mp4",
            gt,
            gt_skel,
            "Ground Truth",
        ),
    ]

    # Tạo hoặc sử dụng lại hai animation.
    for path, poses, skel, title in targets:
        if rerender or not path.is_file():
            render_animation(
                poses,
                skel,
                n,
                fps,
                path,
                title,
            )
        else:
            print(f"  Dùng lại {path.name}")

    # Thứ tự panel:
    # 1. WHAM animation
    # 2. output_wham_video (nếu có)
    # 3. Ground Truth animation
    videos = [targets[0][0]]

    if has_output_wham_video:
        videos.append(output_video)

    videos.append(targets[1][0])

    print(
        f"  {'Có' if has_output_wham_video else 'Không có'} "
        f"output_wham_video -> ghép {len(videos)} video"
    )

    combined = out / "combined_frames.mp4"
    # Ghi lại các video nguồn: đổi output_wham_video (ví dụ trỏ nhầm rồi sửa, hoặc vừa
    # tải từ Drive) thì ghép lại, không dùng nhầm combined_frames.mp4 cũ.
    sources = out / "combined_frames.sources.txt"
    wanted = "\n".join(str(v) for v in videos)
    same_sources = sources.is_file() and sources.read_text(encoding="utf-8") == wanted

    if rerender or not combined.is_file() or not same_sources:
        if combined.is_file() and not rerender:
            print("  Video nguồn đã đổi so với combined_frames.mp4 cũ -> ghép lại")
        combine_videos(
            videos,
            combined,
            fps,
            n,
            has_output_wham_video,
        )
        sources.write_text(wanted, encoding="utf-8")

    return combined


def track_frames(wham_df: pd.DataFrame) -> dict:
    return {int(b): np.sort(g["frame_id"].to_numpy().astype(int))
            for b, g in wham_df.groupby("body_id")}


def process_sequence(seq: dict, manifest: dict, args) -> bool:
    """Trả về False nếu người dùng chọn thoát."""
    if seq.get("skip") is True:
        print(f"\n[SKIP] {seq['name']} (skip=true trong manifest)")
        return True
    
    print(f"\n{'=' * 64}\n{seq['name']}\n{'=' * 64}")
    if (seq["out_dir"] / "mapping.txt").is_file() and not args.force:
        print(f"  Đã có {seq['out_dir'] / 'mapping.txt'} -> bỏ qua (--force để gán lại).")
        return True
    status = sequence_status(seq)
    if status.startswith("THIẾU"):
        print(f"  {status} -> bỏ qua.")
        return True
    seq["out_dir"].mkdir(parents=True, exist_ok=True)
    wham_df = load_wham_csv(seq, manifest["smpl_model"])
    print("  Chuẩn bị Ground Truth ...")
    gt_config = resolve_gt_path(seq)

    print("  Đọc Ground Truth ...")
    gt, gt_skel = load_gt(gt_config)
    video = build_videos(seq, wham_df, gt, gt_skel, args.rerender)
    if not args.no_open:
        open_video(video)
    names = gt_names(gt)
    print_mapping_help(body_id_summary(wham_df), names)
    mapping = ask_mapping(track_frames(wham_df), names, video)
    if mapping is None:
        return False
    write_mapping(seq["out_dir"], mapping, seq)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Gán body_id WHAM với người GT -> mapping.txt")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--split", choices=SPLITS, help="bỏ trống thì chương trình sẽ hỏi")
    p.add_argument("--only", help="chỉ xử lý sequence có tên này")
    p.add_argument("--list", action="store_true", help="chỉ in trạng thái các sequence")
    p.add_argument("--force", action="store_true", help="gán lại kể cả đã có mapping.txt")
    p.add_argument("--rerender", action="store_true", help="vẽ lại video dù đã có")
    p.add_argument("--no-open", action="store_true", help="không tự mở video")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    print(f"Manifest: {args.manifest}")
    print_status(manifest)
    if args.list:
        return
    split = args.split or ask_split()
    seqs = [s for s in manifest.get(split, []) if not args.only or s["name"] == args.only]
    print(f"\nTập {split}: {len(seqs)} sequence")
    for seq in seqs:
        if not process_sequence(seq, manifest, args):
            print("\nĐã thoát. Các mapping đã lưu vẫn giữ nguyên.")
            return
    print("\nXong.")
    print_status(manifest)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng (Ctrl+C).")
