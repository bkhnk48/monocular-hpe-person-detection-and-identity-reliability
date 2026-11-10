"""
Đọc nghiệm luồng thành bảng gán, phân tích hàm mục tiêu, và chấm điểm với
mapping_frames.csv (nhãn gán tay bằng scripts/wham_gt_mapping.py).

Trạng thái mỗi (frame, người GT):
  correct      : chọn đúng một body_id được gán tay là người đó
  correct_miss : chọn missed và frame đó không có body_id nào của người đó
  wrong_id     : chọn một body_id KHÔNG phải người đó (người khác / không phải người)
  missed       : chọn missed trong khi có body_id đúng
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .costs import CostParams
from .network import Network


def assignment_table(net: Network, flow: np.ndarray) -> pd.DataFrame:
    rows = []
    for a in np.flatnonzero(flow):
        n = net.nodes[net.heads[a]]
        if n["kind"] in ("det", "miss"):
            rows.append({"frame": n["frame"], "person": n["person"],
                         "body_id": n["body_id"] if n["kind"] == "det" else np.nan,
                         "pa_mm": n["pa_mm"]})
    df = pd.DataFrame(rows, columns=["frame", "person", "body_id", "pa_mm"])
    df["body_id"] = df["body_id"].astype("Int64")
    return df.sort_values(["frame", "person"]).reset_index(drop=True)


def objective_breakdown(net: Network, flow: np.ndarray, p: CostParams) -> dict:
    used = [net.nodes[net.heads[a]] for a in np.flatnonzero(flow)]
    dets = [n for n in used if n["kind"] == "det"]
    n_det_total = net.n_detections
    kinds = np.asarray(net.arc_kind)[flow == 1]
    arc_costs = np.asarray(net.arc_cost)[flow == 1]
    pose = sum(p.alpha * n["pa_mm"] / p.pose_scale_mm for n in dets)
    n_fn = sum(n["kind"] == "miss" for n in used)
    return {"pose": pose,
            "false_positive": p.c_fp * (n_det_total - len(dets)),
            "false_negative": p.c_fn * n_fn,
            "temporal_and_gaps": float(arc_costs[kinds == "move"].sum()),
            "boundary": float(arc_costs[np.isin(kinds, ["start", "end"])].sum()),
            "n_detections": n_det_total, "n_used": len(dets), "n_missed": int(n_fn)}


def load_truth(path: Path) -> dict:
    """{(frame, người): {body_id,...}} từ mapping_frames.csv."""
    m = pd.read_csv(path, dtype={"person": str}).fillna({"person": ""})
    m = m[m["person"] != ""]
    truth: dict = {}
    for r in m.itertuples(index=False):
        truth.setdefault((int(r.frame), r.person), set()).add(int(r.body_id))
    return truth


def status_of(body_id, truth_ids: set) -> str:
    if pd.isna(body_id):
        return "missed" if truth_ids else "correct_miss"
    return "correct" if int(body_id) in truth_ids else "wrong_id"


def evaluate(assign: pd.DataFrame, truth: dict) -> tuple[pd.DataFrame, dict]:
    df = assign.copy()
    df["truth_body_ids"] = [",".join(map(str, sorted(truth.get((f, k), set()))))
                            for f, k in zip(df["frame"], df["person"])]
    df["status"] = [status_of(b, truth.get((f, k), set()))
                    for f, k, b in zip(df["frame"], df["person"], df["body_id"])]
    counts = df["status"].value_counts().to_dict()
    n = len(df)
    ok = counts.get("correct", 0) + counts.get("correct_miss", 0)
    metrics = {"frames_x_persons": n, "accuracy": ok / n if n else float("nan")}
    for s in ("correct", "correct_miss", "wrong_id", "missed"):
        metrics[s] = int(counts.get(s, 0))
    metrics["id_changes"] = id_changes(df)
    return df, metrics


def id_changes(assign: pd.DataFrame) -> int:
    """Số lần body_id được chọn đổi giữa 2 frame liên tiếp có detection (mọi người)."""
    total = 0
    for _, g in assign.dropna(subset=["body_id"]).groupby("person"):
        b = g.sort_values("frame")["body_id"].to_numpy()
        total += int((b[1:] != b[:-1]).sum())
    return total
