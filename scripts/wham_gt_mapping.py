#!/usr/bin/env python3
"""
Gán body_id của WHAM với người thật trong Ground Truth -> mapping.txt.

Quy trình cho mỗi sequence trong tập được chọn (validation / test) của manifest:
  1. Đã có mapping.txt -> bỏ qua.
  2. pkl WHAM -> RA-WHAM.csv -> video animation WHAM (wham_anim.mp4).
  3. Ground Truth (annot.mat của MPI-INF-3DHP hoặc poses3d/*.npy của Harmony4D)
     -> video animation GT (gt_anim.mp4).
  4. Có video WHAM vẽ đè lên video gốc -> ghép 3 video; không có -> ghép 2 video.
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
import subprocess
import sys
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


def make_skeleton(joints, extra_bones=()):
    """Khung xương = tên khớp + cặp chỉ số xương."""
    pos = {j: i for i, j in enumerate(joints)}
    bones = [(pos[a], pos[b]) for a, b in list(LIMB_BONES) + list(extra_bones)]
    top = [pos["left_shoulder"], pos["right_shoulder"]]
    bottom = [pos["left_ankle"], pos["right_ankle"]]
    return {"joints": joints, "bones": bones, "top": top, "bottom": bottom}


SKELETONS = {
    "wham": make_skeleton(["pelvis"] + LIMBS),
    "mpi13": make_skeleton(list(MPI_IDX),
                           [("neck", "left_shoulder"), ("neck", "right_shoulder")]),
    "coco17": make_skeleton(COCO17, [("nose", "left_eye"), ("nose", "right_eye"),
                                     ("left_eye", "left_ear"), ("right_eye", "right_ear"),
                                     ("nose", "left_shoulder"), ("nose", "right_shoulder")]),
}

PALETTE = [(40, 40, 230), (230, 150, 0), (40, 180, 40), (200, 40, 200), (0, 200, 230),
           (0, 120, 255), (160, 160, 0), (120, 40, 160), (80, 80, 80), (40, 200, 160)]

# ============================================================
# 2. MANIFEST
# ============================================================


def load_manifest(path: Path) -> dict:
    """Đọc manifest, đổi mọi đường dẫn thành tuyệt đối."""
    m = json.loads(path.read_text(encoding="utf-8"))
    base = (path.parent / m.get("base_dir", ".")).resolve()
    m["smpl_model"] = base / m["smpl_model"]
    m["output_dir"] = base / m["output_dir"]
    for split in SPLITS:
        for seq in m.get(split, []):
            seq["wham_pkl"] = base / seq["wham_pkl"]
            seq["gt"]["path"] = base / seq["gt"]["path"]
            ov = seq.get("overlay_video") or ""
            seq["overlay_video"] = base / ov if ov else None
            seq["out_dir"] = m["output_dir"] / seq["name"]
    return m


def sequence_status(seq: dict) -> str:
    if (seq["out_dir"] / "mapping.txt").is_file():
        return "đã có mapping"
    missing = [k for k, p in (("pkl", seq["wham_pkl"]), ("GT", seq["gt"]["path"]))
               if not p.exists()]
    return "THIẾU " + ", ".join(missing) if missing else "chưa gán"


def print_status(manifest: dict):
    for split in SPLITS:
        print(f"\n[{split}]")
        for seq in manifest.get(split, []):
            ov = "3 video" if seq["overlay_video"] and seq["overlay_video"].is_file() else "2 video"
            print(f"  {seq['name']:<28} {ov:<8} {sequence_status(seq)}")


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


def wham_pkl_to_csv(pkl: Path, smpl: Path, out_csv: Path) -> pd.DataFrame:
    if out_csv.is_file():
        print(f"  Dùng lại CSV: {out_csv.name}")
        return pd.read_csv(out_csv)
    print(f"  Đọc pkl WHAM: {pkl}")
    results = joblib.load(pkl)
    J = load_j_regressor(smpl)
    tables = [wham_subject_table(sid, d, J) for sid, d in results.items()
              if len(d.get("frame_ids", []))]
    df = pd.concat(tables, ignore_index=True).sort_values(["frame_id", "body_id"])
    df.to_csv(out_csv, index=False)
    print(f"  Đã lưu {len(df)} dòng, {df['body_id'].nunique()} body_id -> {out_csv.name}")
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


def mpi_gt_poses(mat: Path, camera: int, person: str) -> dict:
    """annot.mat: annot3[camera] (F, 28*3) mm, hệ toạ độ camera -> 13 khớp, mét."""
    data = loadmat(mat, squeeze_me=True, struct_as_record=False)
    ann = np.asarray(data["annot3"][camera], dtype=float)
    ann = ann.reshape(len(ann), -1, 3)[:, list(MPI_IDX.values())] / 1000.0
    return {f: {person: ann[f]} for f in range(len(ann)) if np.isfinite(ann[f]).all()}


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
        return mpi_gt_poses(gt["path"], int(gt["camera"]), str(gt["person"])), "mpi13"
    if gt["type"] == "harmony4d":
        return h4d_gt_poses(gt["path"]), "coco17"
    raise ValueError(f"gt.type không hỗ trợ: {gt['type']}")


def gt_names(poses: dict) -> list[str]:
    return sorted({p for people in poses.values() for p in people})

# ============================================================
# 5. VẼ ANIMATION (OpenCV, nhanh hơn matplotlib nhiều lần)
# ============================================================


def up_rotation(poses: dict, skel: dict) -> np.ndarray:
    """Ma trận xoay đưa hướng 'vai - mắt cá' trung bình về trục +z."""
    vecs = [p[skel["top"]].mean(0) - p[skel["bottom"]].mean(0)
            for people in poses.values() for p in people.values()]
    up = np.nanmean(np.array(vecs), axis=0)
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if c < -0.999:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
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
# 6. VIDEO: FPS, GHÉP, MỞ
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


def combine_videos(videos: list[Path], out: Path, fps: float):
    """Ghép ngang 2-3 video, cùng chiều cao; video ngắn hơn giữ frame cuối."""
    dur = max(video_info(v)[1] for v in videos)
    parts = [f"[{i}:v]scale=-2:{VIDEO_H},fps={fps},"
             f"tpad=stop_mode=clone:stop_duration={dur + 1:.2f}[v{i}]"
             for i in range(len(videos))]
    stack = "".join(f"[v{i}]" for i in range(len(videos)))
    filt = ";".join(parts) + f";{stack}hstack=inputs={len(videos)}[out]"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-stats"]
    for v in videos:
        cmd += ["-i", str(v)]
    cmd += ["-filter_complex", filt, "-map", "[out]", "-t", f"{dur:.3f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    print(f"  Ghép {len(videos)} video -> {out.name}")
    subprocess.run(cmd, check=True)


def open_video(path: Path):
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        print(f"  Hãy mở video: {path}")

# ============================================================
# 7. HỎI NGƯỜI DÙNG VÀ GHI MAPPING
# ============================================================


def print_mapping_help(summary: pd.DataFrame, names: list[str]):
    print("\n  body_id   frame đầu - cuối   số frame")
    for bid, r in summary.iterrows():
        print(f"  {bid:>7}   {r['first']:>6} - {r['last']:<6}   {r['frames']:>7}")
    print("\n  Người trong Ground Truth:")
    for i, n in enumerate(names, 1):
        print(f"    {i}. {n}")
    print("\n  Nhập số thứ tự hoặc tên người GT. Enter = không phải người GT.")
    print("  v = mở lại video, b = quay lại body_id trước, q = thoát không lưu.")


def parse_answer(ans: str, names: list[str]):
    """Trả về tên GT, '' (không phải người), hoặc lệnh 'v'/'b'/'q'; None nếu sai."""
    if ans in ("", "v", "b", "q"):
        return ans
    if ans.isdigit() and 1 <= int(ans) <= len(names):
        return names[int(ans) - 1]
    return ans if ans in names else None


def ask_mapping(summary: pd.DataFrame, names: list[str], video: Path):
    """Hỏi lần lượt từng body_id. Trả về {body_id: tên GT hoặc ''}, None nếu thoát."""
    ids, mapping, i = list(summary.index), {}, 0
    while i < len(ids):
        try:
            raw = input(f"  body_id {ids[i]} -> ").strip()
        except EOFError:
            return None
        ans = parse_answer(raw, names)
        if ans is None:
            print("    Không hợp lệ, nhập lại.")
        elif ans == "q":
            return None
        elif ans == "v":
            open_video(video)
        elif ans == "b":
            i = max(0, i - 1)
        else:
            mapping[ids[i]] = ans
            i += 1
    return mapping


def confirm_mapping(mapping: dict) -> bool:
    print("\n  Kết quả:")
    for bid, name in mapping.items():
        print(f"    {bid} = {name or '(không phải người GT)'}")
    return input("  Lưu mapping.txt? [Y/n]: ").strip().lower() in ("", "y", "yes", "c", "co")


def write_mapping(path: Path, mapping: dict, seq: dict):
    lines = ["# Ánh xạ WHAM body_id = người trong Ground Truth",
             f"# sequence: {seq['name']}",
             f"# wham_pkl: {seq['wham_pkl']}",
             f"# gt: {seq['gt']['path']}",
             "# body_id không có dòng gán (dòng # ...) = không phải người GT"]
    lines += [f"{b} = {n}" for b, n in mapping.items() if n]
    lines += [f"# {b} = (không phải người GT)" for b, n in mapping.items() if not n]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Đã lưu: {path}")

# ============================================================
# 8. XỬ LÝ MỘT SEQUENCE VÀ CHƯƠNG TRÌNH CHÍNH
# ============================================================


def build_videos(seq: dict, wham_df: pd.DataFrame, gt: dict, gt_skel: str,
                 rerender: bool) -> Path:
    out, overlay = seq["out_dir"], seq["overlay_video"]
    has_overlay = overlay is not None and overlay.is_file()
    fps = video_info(overlay)[0] if has_overlay else float(seq.get("fps", 25))
    n = max(int(wham_df["frame_id"].max()) + 1, max(gt) + 1)
    targets = [(out / "wham_anim.mp4", wham_poses(wham_df), "wham", "WHAM"),
               (out / "gt_anim.mp4", gt, gt_skel, "Ground Truth")]
    for path, poses, skel, title in targets:
        if rerender or not path.is_file():
            render_animation(poses, skel, n, fps, path, title)
        else:
            print(f"  Dùng lại {path.name}")
    videos = ([overlay] if has_overlay else []) + [t[0] for t in targets]
    print(f"  {'Có' if has_overlay else 'Không có'} video overlay WHAM -> ghép {len(videos)} video")
    combined = out / "combined.mp4"
    if rerender or not combined.is_file():
        combine_videos(videos, combined, fps)
    return combined


def process_sequence(seq: dict, manifest: dict, args) -> bool:
    """Trả về False nếu người dùng chọn thoát."""
    print(f"\n{'=' * 64}\n{seq['name']}\n{'=' * 64}")
    mapping_path = seq["out_dir"] / "mapping.txt"
    if mapping_path.is_file() and not args.force:
        print(f"  Đã có {mapping_path} -> bỏ qua (dùng --force để gán lại).")
        return True
    status = sequence_status(seq)
    if status.startswith("THIẾU"):
        print(f"  {status} -> bỏ qua.")
        return True
    seq["out_dir"].mkdir(parents=True, exist_ok=True)
    wham_df = wham_pkl_to_csv(seq["wham_pkl"], manifest["smpl_model"],
                              seq["out_dir"] / "RA-WHAM.csv")
    print("  Đọc Ground Truth ...")
    gt, gt_skel = load_gt(seq["gt"])
    video = build_videos(seq, wham_df, gt, gt_skel, args.rerender)
    if not args.no_open:
        open_video(video)
    summary = body_id_summary(wham_df)
    names = gt_names(gt)
    print_mapping_help(summary, names)
    mapping = ask_mapping(summary, names, video)
    if mapping is None:
        return False
    if confirm_mapping(mapping):
        write_mapping(mapping_path, mapping, seq)
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
