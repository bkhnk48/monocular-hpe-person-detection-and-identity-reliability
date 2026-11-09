#!/usr/bin/env python3
"""
Khảo sát "thông tin riêng của track" (identity signature) cho câu hỏi:
  MC-MCF có cơ sở để khác SC-MCF không?

Hai loại signature:
  1. Bone-length signature  : tính từ 13 khớp trong WHAM CSV (luôn có).
  2. SMPL betas (10 chiều)  : từ WHAM PKL (nếu cung cấp --wham-pkl).

Mỗi signature được đánh giá theo 4 câu hỏi (H1–H4):
  H1  Ổn định theo thời gian trong một track?        (không cần nhãn)
  H2  Profile tích lũy từ k detection đầu có dự báo   (không cần nhãn)
      được phần còn lại của track không?
  H3  Phân biệt được hai người KHÁC nhau không?      (cần nhãn identity, >=2 người
                                                      -> Harmony4D)
  H4  Detection ghost có lệch khỏi profile của        (cần nhãn real/ghost
      người thật không? Các fragment của cùng           -> MPI-INF-3DHP dùng được)
      một người có cùng profile không?

LƯU Ý PHƯƠNG PHÁP: nhãn GT chỉ dùng để PHÂN TÍCH đặc trưng (feature analysis),
không đưa vào solver. Nên báo cáo phần này như một nghiên cứu chẩn đoán riêng.

Ví dụ (MPI-INF-3DHP, dùng output matching sẵn có để tạo nhãn real/ghost):
  python scripts/survey_identity_signature.py \
      --wham-csv data/raw/mpi_inf_3dhp/wham_predictions/RA-WHAM_S8_Seq1_video_cam0.csv \
      --mpi-best-detection outputs/subject_matching/best_detection_per_frame.csv \
      --out outputs/identity_signature/S8_Seq1

Ví dụ (Harmony4D; nhãn tạo bởi scripts/label_harmony4d_wham.py, chỉ cần PKL):
  python scripts/survey_identity_signature.py \
      --wham-pkl <wham_output.pkl> \
      --labels outputs/harmony4d/041_grappling2_cam01/labels.csv --exclude-ambiguous \
      --out outputs/identity_signature/041_grappling2_cam01

File nhãn: cột frame_id, body_id, identity [, ambiguous]. identity = mã người GT
("aria01", "P0", ...) hoặc "unmatched"/"ghost" nếu không khớp người GT nào.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
NON_ID = {"ghost", "unmatched"}  # nhãn không phải người GT

# ---------------------------------------------------------------------------
# 1. Signature: bone lengths
# ---------------------------------------------------------------------------
# Gộp trái/phải để giảm nhiễu. Mỗi bone = trung bình độ dài 2 bên (nếu có).
BONES = {
    "shoulder_width": [("left_shoulder", "right_shoulder")],
    "hip_width": [("left_hip", "right_hip")],
    "torso": [("pelvis", "mid_shoulder")],
    "upper_arm": [("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")],
    "forearm": [("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")],
    "thigh": [("left_hip", "left_knee"), ("right_hip", "right_knee")],
    "shin": [("left_knee", "left_ankle"), ("right_knee", "right_ankle")],
}


def _joint(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "mid_shoulder":
        return 0.5 * (_joint(df, "left_shoulder") + _joint(df, "right_shoulder"))
    cols = [f"{name}_{a}" for a in "xyz"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột khớp: {missing}")
    return df[cols].to_numpy(dtype=float)


def bone_signature(df: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    for bone, pairs in BONES.items():
        lens = [np.linalg.norm(_joint(df, a) - _joint(df, b), axis=1) for a, b in pairs]
        feats[bone] = np.nanmean(np.vstack(lens), axis=0)
    abs_df = pd.DataFrame(feats, index=df.index)
    # Tỉ lệ (scale-invariant): quan trọng vì monocular có mơ hồ tỉ lệ.
    ratio_df = abs_df.div(abs_df.sum(axis=1), axis=0).add_prefix("ratio_")
    return pd.concat([abs_df.add_prefix("len_"), ratio_df], axis=1)


# ---------------------------------------------------------------------------
# 2. Signature: SMPL betas từ WHAM PKL
# ---------------------------------------------------------------------------
def load_wham_betas(path: Path) -> pd.DataFrame:
    """
    Đọc output WHAM dạng dict {subject_id: {'betas': (N,10), 'frame_ids': (N,), ...}}.
    Giả định subject_id trong PKL trùng body_id trong CSV — cần kiểm tra lại.
    """
    try:
        import joblib

        data = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            data = pickle.load(f)

    rows = []
    for sid, res in data.items():
        if not isinstance(res, dict) or "betas" not in res:
            continue
        betas = np.asarray(res["betas"], dtype=float).reshape(-1, 10)
        frames = np.asarray(res.get("frame_ids", np.arange(len(betas)))).reshape(-1)
        n = min(len(betas), len(frames))
        part = pd.DataFrame(betas[:n], columns=[f"beta_{k}" for k in range(10)])
        part.insert(0, "frame_id", frames[:n].astype(int))
        part.insert(1, "body_id", int(sid) if str(sid).lstrip("-").isdigit() else sid)
        rows.append(part)
    if not rows:
        raise ValueError(f"Không tìm thấy key 'betas' trong {path}")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Nhãn
# ---------------------------------------------------------------------------
def labels_from_mpi_best(path: Path, wham: pd.DataFrame, thresh_mm: float) -> pd.DataFrame:
    """
    MPI-INF-3DHP chỉ có 1 người GT. Detection được gán 'P0' nếu là detection
    tốt nhất của frame và PA-MPJPE < thresh_mm; còn lại gán 'ghost'.
    """
    b = pd.read_csv(path)
    best = b.dropna(subset=["best_body_id"])
    real = best[best["best_pa_mpjpe_mm"] < thresh_mm]
    real_keys = set(zip(real["frame_id"].astype(int), real["best_body_id"].astype(int)))
    lab = wham[["frame_id", "body_id"]].copy()
    lab["identity"] = [
        "P0" if (int(f), int(bid)) in real_keys else "ghost"
        for f, bid in zip(lab["frame_id"], lab["body_id"])
    ]
    return lab


# ---------------------------------------------------------------------------
# 4. Thống kê
# ---------------------------------------------------------------------------
def robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) qua rank (Mann–Whitney)."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pooled_within_scale(df: pd.DataFrame, feats: list[str], group: str) -> pd.Series:
    """Độ lệch chuẩn (robust) trong-nhóm, gộp trung vị qua các nhóm."""
    per = df.groupby(group)[feats].agg(lambda s: robust_std(s.to_numpy()))
    scale = per.median()
    return scale.replace(0, np.nan).fillna(df[feats].std())


def standardized_dist(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Khoảng cách Euclid trên đặc trưng đã chuẩn hóa, chia sqrt(dim)."""
    z = (a - b) / scale
    return np.sqrt(np.nanmean(z**2, axis=-1))


def h1_stability(df, feats, out: Path, tag: str) -> pd.DataFrame:
    rows = []
    for bid, g in df.sort_values("frame_id").groupby("body_id"):
        if len(g) < 10:
            continue
        half = len(g) // 2
        for f in feats:
            x = g[f].to_numpy()
            med = np.nanmedian(x)
            rs = robust_std(x)
            jump = np.nanmedian(np.abs(np.diff(x)))
            drift = abs(np.nanmedian(x[:half]) - np.nanmedian(x[half:]))
            rows.append(
                dict(body_id=bid, n=len(g), feature=f, median=med, robust_std=rs,
                     cv_pct=100 * rs / abs(med) if med else np.nan,
                     frame_to_frame_jump=jump,
                     drift_halves=drift,
                     drift_over_std=drift / rs if rs else np.nan)
            )
    res = pd.DataFrame(rows)
    res.to_csv(out / f"H1_stability_{tag}.csv", index=False)
    return res


def h2_warmup(df, feats, scale, out: Path, tag: str, ks=(5, 10, 30, 60)) -> pd.DataFrame:
    rows = []
    for bid, g in df.sort_values("frame_id").groupby("body_id"):
        X = g[feats].to_numpy()
        full = np.nanmedian(X, axis=0)
        for k in ks:
            if len(X) <= k + 10:
                continue
            prof = np.nanmedian(X[:k], axis=0)
            d_rest = standardized_dist(X[k:], prof, scale.to_numpy())
            rows.append(dict(body_id=bid, k=k,
                             profile_err_vs_full=float(standardized_dist(prof, full, scale.to_numpy())),
                             median_dist_rest=float(np.nanmedian(d_rest))))
    res = pd.DataFrame(rows)
    res.to_csv(out / f"H2_profile_warmup_{tag}.csv", index=False)
    return res


def track_profile_matrix(df, feats, scale, out: Path, tag: str) -> pd.DataFrame:
    prof = df.groupby("body_id")[feats].median()
    ids = prof.index.tolist()
    M = np.zeros((len(ids), len(ids)))
    for a, ia in enumerate(ids):
        for b, ib in enumerate(ids):
            M[a, b] = standardized_dist(prof.loc[ia].to_numpy(), prof.loc[ib].to_numpy(), scale.to_numpy())
    mat = pd.DataFrame(M, index=ids, columns=ids)
    mat.to_csv(out / f"track_profile_distance_{tag}.csv")
    return mat


def sample_pairs(idx_a: np.ndarray, idx_b: np.ndarray, n: int, same: bool):
    if same:
        if len(idx_a) < 2:
            return np.empty((0, 2), int)
        i = RNG.choice(idx_a, n)
        j = RNG.choice(idx_a, n)
        keep = i != j
        return np.stack([i[keep], j[keep]], 1)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.empty((0, 2), int)
    return np.stack([RNG.choice(idx_a, n), RNG.choice(idx_b, n)], 1)


def h3_h4_labeled(df, feats, scale, out: Path, tag: str, min_gap: int) -> dict:
    """
    H3: same-identity vs different-identity (cần >=2 người thật).
    H4a: real vs ghost — khoảng cách tới profile của người thật gần nhất.
    H4b: fragment consistency — các body_id khác nhau nhưng cùng identity.
    Profile tính leave-track-out để tránh tự so với chính mình.
    """
    res: dict = {}
    X = df[feats].to_numpy()
    sc = scale.to_numpy()
    real = df[~df["identity"].isin(NON_ID)]
    ids = sorted(real["identity"].unique())

    # ---- H3 ---------------------------------------------------------------
    # Tách cặp cùng người thành: cùng track (đo ổn định) và KHÁC track (đo khả năng
    # nối lại fragment / re-identification). Cặp khác track mới là câu hỏi cho MC-MCF.
    if len(ids) >= 2:
        R = real.reset_index(drop=True)
        XR = R[feats].to_numpy()
        idn = R["identity"].to_numpy()
        bid = R["body_id"].to_numpy()
        fr = R["frame_id"].to_numpy()
        n = len(R)
        m = min(200000, n * (n - 1) // 2)
        i = RNG.integers(0, n, m)
        j = RNG.integers(0, n, m)
        keep = (i != j) & ((idn[i] != idn[j]) | (np.abs(fr[i] - fr[j]) >= min_gap))
        i, j = i[keep], j[keep]
        d = standardized_dist(XR[i], XR[j], sc)
        same = idn[i] == idn[j]
        same_track = same & (bid[i] == bid[j])
        cross_track = same & (bid[i] != bid[j])
        diff = ~same
        res["H3_n_pairs_same_track/cross_track/diff"] = f"{same_track.sum()}/{cross_track.sum()}/{diff.sum()}"
        res["H3_median_same_track"] = float(np.nanmedian(d[same_track])) if same_track.any() else np.nan
        res["H3_median_same_cross_track"] = float(np.nanmedian(d[cross_track])) if cross_track.any() else np.nan
        res["H3_median_diff"] = float(np.nanmedian(d[diff]))
        res["H3_auc_diff_vs_same_track"] = auc(d[diff], d[same_track])
        if cross_track.any():
            res["H3_auc_diff_vs_same_cross_track (re-ID, quyết định cho MC-MCF)"] = auc(d[diff], d[cross_track])
        else:
            res["H3_note_cross_track"] = "Không có người nào có >=2 track -> chưa đo được re-ID."
        fr_rows = []
        for f in feats:
            gm = real.groupby("identity")[f]
            between = gm.median().var()
            within = gm.agg(lambda s_: robust_std(s_.to_numpy()) ** 2).mean()
            fr_rows.append(dict(feature=f, fisher_ratio=between / within if within else np.nan))
        pd.DataFrame(fr_rows).to_csv(out / f"H3_fisher_ratio_{tag}.csv", index=False)
        np.savez(out / f"H3_pair_dists_{tag}.npz", same_track=d[same_track],
                 cross_track=d[cross_track], diff=d[diff])
    else:
        res["H3_note"] = "Chỉ có 1 người thật -> không đánh giá được phân biệt giữa người (cần Harmony4D)."

    # ---- H4a: real vs ghost ----------------------------------------------
    ghost = df[df["identity"].isin(NON_ID)]
    if len(ghost) and len(real):
        d_real, d_ghost = [], []
        for (bid, p), g in real.groupby(["body_id", "identity"]):
            others = real[(real["identity"] == p) & (real["body_id"] != bid)]
            ref = others if len(others) >= 20 else real[real["identity"] == p]
            prof = np.nanmedian(ref[feats].to_numpy(), axis=0)
            d_real.append(standardized_dist(g[feats].to_numpy(), prof, sc))
        profs = np.stack([np.nanmedian(real[real["identity"] == p][feats].to_numpy(), axis=0) for p in ids])
        G = ghost[feats].to_numpy()
        dg = np.min(np.stack([standardized_dist(G, pr, sc) for pr in profs]), axis=0)
        d_ghost.append(dg)
        d_real = np.concatenate(d_real)
        d_ghost = np.concatenate(d_ghost)
        res["H4a_auc_ghost_vs_real"] = auc(d_ghost, d_real)
        res["H4a_median_real"] = float(np.nanmedian(d_real))
        res["H4a_median_ghost"] = float(np.nanmedian(d_ghost))
        res["H4a_n_real"], res["H4a_n_ghost"] = int(len(d_real)), int(len(d_ghost))
        np.savez(out / f"H4a_dists_{tag}.npz", real=d_real, ghost=d_ghost)

    # ---- H4b: fragment consistency ---------------------------------------
    frag_rows = []
    for p in ids:
        sub = real[real["identity"] == p]
        frags = sub.groupby("body_id")
        prof = {b: np.nanmedian(g[feats].to_numpy(), axis=0) for b, g in frags if len(g) >= 20}
        keys = sorted(prof)
        for a_i, a in enumerate(keys):
            for b in keys[a_i + 1:]:
                frag_rows.append(dict(identity=p, body_id_a=a, body_id_b=b,
                                      profile_dist=float(standardized_dist(prof[a], prof[b], sc))))
    if frag_rows:
        fr = pd.DataFrame(frag_rows)
        fr.to_csv(out / f"H4b_fragment_consistency_{tag}.csv", index=False)
        res["H4b_median_fragment_profile_dist"] = float(fr["profile_dist"].median())
    return res


# ---------------------------------------------------------------------------
# 5. Plot (tùy chọn)
# ---------------------------------------------------------------------------
def try_plot(df, feats, out: Path, tag: str, labeled: bool):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = len(feats)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, feats):
        for bid, g in df.groupby("body_id"):
            ax.plot(g["frame_id"], g[f], ".", ms=1.5, label=str(bid))
        if labeled:
            gh = df[df["identity"].isin(NON_ID)]
            ax.plot(gh["frame_id"], gh[f], "x", ms=2, color="k", alpha=0.3)
        ax.set_ylabel(f, fontsize=7)
    axes[0].legend(ncol=12, fontsize=6, markerscale=5, title="body_id (x = ghost)", title_fontsize=6)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out / f"signature_timeline_{tag}.png", dpi=130)
    plt.close(fig)

    if labeled and (out / f"H4a_dists_{tag}.npz").exists():
        z = np.load(out / f"H4a_dists_{tag}.npz")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bins = np.linspace(0, np.nanpercentile(np.concatenate([z["real"], z["ghost"]]), 99), 60)
        ax.hist(z["real"], bins, alpha=0.6, density=True, label="real")
        ax.hist(z["ghost"], bins, alpha=0.6, density=True, label="ghost")
        ax.set_xlabel("distance to identity profile (standardized)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"H4a_hist_{tag}.png", dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def track_relation_table(df, feats, scale, out: Path, tag: str, min_purity: float) -> tuple[pd.DataFrame, dict]:
    """
    So sánh prototype (median) giữa mọi cặp body_id, phân loại quan hệ theo
    identity chiếm đa số của mỗi track:
      same_person      : hai fragment của cùng một người GT  -> khoảng cách nên NHỎ
      different_person : hai người GT khác nhau              -> khoảng cách nên LỚN
      involves_unmatched / mixed_track : báo cáo riêng, không tính vào AUC.
    Đây là câu hỏi trực tiếp cho MC-MCF: profile track có nối đúng fragment và
    tách đúng người không?
    """
    info = []
    for bid, g in df.groupby("body_id"):
        vc = g["identity"].value_counts()
        info.append(dict(body_id=bid, n=len(g), dominant=vc.index[0], purity=vc.iloc[0] / len(g),
                         proto=np.nanmedian(g[feats].to_numpy(), axis=0)))
    rows = []
    for a, b in itertools.combinations(info, 2):
        if a["purity"] < min_purity or b["purity"] < min_purity:
            rel = "mixed_track"
        elif a["dominant"] in NON_ID or b["dominant"] in NON_ID:
            rel = "involves_unmatched"
        elif a["dominant"] == b["dominant"]:
            rel = "same_person"
        else:
            rel = "different_person"
        rows.append(dict(body_a=a["body_id"], body_b=b["body_id"],
                         id_a=a["dominant"], id_b=b["dominant"], n_a=a["n"], n_b=b["n"],
                         relation=rel,
                         dist_raw=float(np.linalg.norm(a["proto"] - b["proto"])),
                         dist_std=float(standardized_dist(a["proto"], b["proto"], scale.to_numpy()))))
    tab = pd.DataFrame(rows).sort_values("dist_raw")
    tab.to_csv(out / f"H3_track_pairs_{tag}.csv", index=False)
    res = {}
    same = tab.loc[tab.relation == "same_person", "dist_raw"].to_numpy()
    diff = tab.loc[tab.relation == "different_person", "dist_raw"].to_numpy()
    res["H3track_n_same_pairs"] = int(len(same))
    res["H3track_n_diff_pairs"] = int(len(diff))
    if len(same) and len(diff):
        res["H3track_auc_diff_vs_same"] = auc(diff, same)
        res["H3track_max_same"] = float(same.max())
        res["H3track_min_diff"] = float(diff.min())
    return tab, res


def run_one(df, feats, tag, out, labeled, min_gap, min_purity=0.8) -> list[str]:
    df = df.dropna(subset=feats)
    lines = [f"\n## Signature: `{tag}` ({len(feats)} chiều, {len(df)} detection)\n"]
    scale = pooled_within_scale(df, feats, "body_id")

    h1 = h1_stability(df, feats, out, tag)
    if len(h1):
        s = h1.groupby("feature")[["cv_pct", "drift_over_std"]].median().round(2)
        lines.append("**H1 – Ổn định trong track** (trung vị qua các track; CV% thấp và drift/std < 1 là tốt):\n")
        lines.append(s.to_markdown())

    h2 = h2_warmup(df, feats, scale, out, tag)
    if len(h2):
        s = h2.groupby("k")[["profile_err_vs_full", "median_dist_rest"]].median().round(3)
        lines.append("\n**H2 – Profile từ k detection đầu** (sai số so với profile toàn track, đơn vị = within-std):\n")
        lines.append(s.to_markdown())

    mat = track_profile_matrix(df, feats, scale, out, tag)
    lines.append("\n**Khoảng cách profile giữa các body_id** (đơn vị within-std; >~2 là tách biệt rõ):\n")
    lines.append(mat.round(2).to_markdown())

    if labeled:
        r = h3_h4_labeled(df, feats, scale, out, tag, min_gap)
        lines.append("\n**H3/H4 – Có nhãn:**\n")
        for k, v in r.items():
            lines.append(f"- `{k}`: {round(v, 3) if isinstance(v, float) else v}")
        tab, rt = track_relation_table(df, feats, scale, out, tag, min_purity)
        lines.append("\n**H3 mức track – khoảng cách prototype theo quan hệ** (Euclid thô):\n")
        lines.append(tab.groupby("relation")["dist_raw"].agg(["count", "median", "min", "max"]).round(3).to_markdown())
        for k, v in rt.items():
            lines.append(f"- `{k}`: {round(v, 3) if isinstance(v, float) else v}")
        lines.append("\n" + tab.drop(columns=["dist_std"]).round(3).to_markdown(index=False))
        lines.append("\n_AUC ≈ 0.5: không phân biệt; ≥ 0.8: đặc trưng hữu ích; ≥ 0.9: mạnh._")

    try_plot(df, feats, out, tag, labeled)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wham-csv", type=Path, help="WHAM CSV 13 khớp (để tính bone length)")
    ap.add_argument("--wham-pkl", type=Path, help="WHAM output PKL có 'betas'")
    ap.add_argument("--labels", type=Path, help="CSV: frame_id, body_id, identity [, ambiguous]")
    ap.add_argument("--mpi-best-detection", type=Path,
                    help="outputs/subject_matching/best_detection_per_frame.csv (MPI, 1 người)")
    ap.add_argument("--real-thresh-mm", type=float, default=150.0,
                    help="Ngưỡng PA-MPJPE để coi detection là người thật (MPI)")
    ap.add_argument("--exclude-ambiguous", action="store_true",
                    help="Bỏ detection có cột ambiguous=1 (chồng lấn nhiều người GT)")
    ap.add_argument("--warmup-frames", type=int, default=0,
                    help="Bỏ N frame đầu của mỗi track (giai đoạn khởi động WHAM)")
    ap.add_argument("--min-purity", type=float, default=0.8,
                    help="Track có tỉ lệ identity chiếm đa số < ngưỡng bị coi là mixed_track")
    ap.add_argument("--min-frame-gap", type=int, default=30,
                    help="Cặp same-identity phải cách nhau >= N frame (tránh tương quan tạm thời)")
    ap.add_argument("--out", type=Path, default=Path("outputs/identity_signature"))
    args = ap.parse_args()
    if not args.wham_csv and not args.wham_pkl:
        ap.error("Cần ít nhất --wham-csv hoặc --wham-pkl")
    args.out.mkdir(parents=True, exist_ok=True)

    tables = {}  # tag -> (DataFrame, feature list)
    if args.wham_csv:
        wham = pd.read_csv(args.wham_csv)
        wham["frame_id"] = wham["frame_id"].astype(int)
        sig = pd.concat([wham[["frame_id", "body_id"]], bone_signature(wham)], axis=1)
        tables["bone_len"] = (sig, [c for c in sig.columns if c.startswith("len_")])
        tables["bone_ratio"] = (sig, [c for c in sig.columns if c.startswith("ratio_")])
        keys = sig[["frame_id", "body_id"]]
    if args.wham_pkl:
        betas = load_wham_betas(args.wham_pkl)
        tables["betas"] = (betas, [f"beta_{k}" for k in range(10)])
        if not args.wham_csv:
            keys = betas[["frame_id", "body_id"]]

    labeled = False
    if args.labels:
        lab = pd.read_csv(args.labels)
        if args.exclude_ambiguous and "ambiguous" in lab.columns:
            lab = lab[lab["ambiguous"] == 0]
        lab = lab[["frame_id", "body_id", "identity"]]
        labeled = True
    elif args.mpi_best_detection:
        if not args.wham_csv:
            ap.error("--mpi-best-detection cần --wham-csv")
        lab = labels_from_mpi_best(args.mpi_best_detection, wham, args.real_thresh_mm)
        labeled = True

    report = ["# Khảo sát identity signature cho MC-MCF"]
    for name in ("wham_csv", "wham_pkl", "labels"):
        if getattr(args, name):
            report.append(f"- {name}: `{getattr(args, name)}`")
    report.append(f"- Số detection: {len(keys)}, số body_id: {keys['body_id'].nunique()}")
    if labeled:
        lab["identity"] = lab["identity"].astype(str)
        report.append(f"- Nhãn: {lab['identity'].value_counts().to_dict()}"
                      + (" (đã bỏ ambiguous)" if args.exclude_ambiguous else ""))
        lab.to_csv(args.out / "labels_used.csv", index=False)

    for tag, (df, feats) in tables.items():
        df = df.copy()
        df["frame_id"] = df["frame_id"].astype(int)
        df["body_id"] = df["body_id"].astype(int)
        if args.warmup_frames:
            first = df.groupby("body_id")["frame_id"].transform("min")
            df = df[df["frame_id"] >= first + args.warmup_frames]
        if labeled:
            how = "inner" if args.labels else "left"
            df = df.merge(lab, on=["frame_id", "body_id"], how=how)
            df["identity"] = df["identity"].fillna("ghost")
            if len(df) == 0:
                report.append(f"\n**CẢNH BÁO [{tag}]:** không khớp (frame_id, body_id) với file nhãn.")
                continue
        report += run_one(df, feats, tag, args.out, labeled, args.min_frame_gap, args.min_purity)

    (args.out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n-> Kết quả: {args.out.resolve()}")


if __name__ == "__main__":
    main()
