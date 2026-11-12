"""
Chấm điểm nghiệm gán với mapping_frames.csv (nhãn gán tay bằng scripts/wham_gt_mapping.py).

Có hai định dạng nhãn, KHÔNG dùng lẫn:
  load_truth()            {(frame, body_id): người hoặc None}
                          -> evaluate_assignment(), macro_average() (pose_assoc.compare)
  load_truth_by_person()  {(frame, người): {body_id, ...}}
                          -> evaluate() (build_network), node_status() (tune)

Trạng thái mỗi (frame, người GT) trong evaluate():
  correct      : chọn đúng một body_id được gán tay là người đó
  correct_miss : chọn missed và frame đó không có body_id nào của người đó
  wrong_id     : chọn một body_id KHÔNG phải người đó (người khác / không phải người)
  missed       : chọn missed trong khi có body_id đúng
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .costs import CostParams
from .network import Network


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _normalise_person(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in ("", "nan", "none", "null", "na"):
        return None
    return value


def load_truth(mapping_csv: str | None = None, mapping_df=None):
    """
    Đọc mapping_frames.csv.

    Kết quả:
      truth[(frame, body_id)] = person hoặc None

    Dòng có person rỗng được giữ lại với giá trị None:
    detection đó đã được gán nhãn là không thuộc một GT person.
    """
    if mapping_df is None:
        if mapping_csv is None:
            raise ValueError("Cần mapping_csv hoặc mapping_df.")
        mapping_df = pd.read_csv(mapping_csv)

    required = {"frame", "body_id", "person"}
    missing = required - set(mapping_df.columns)
    if missing:
        raise ValueError(
            f"mapping_frames.csv thiếu các cột: {sorted(missing)}"
        )

    truth = {}

    for row in mapping_df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        body_id = getattr(row, "body_id")
        person = _normalise_person(getattr(row, "person"))
        truth[(frame, body_id)] = person

    return truth


def _node_maps(net):
    id_to_node = {}
    node_index_by_id = {}

    for i, node in enumerate(net.nodes):
        node_id = int(_get(node, "id", i))
        id_to_node[node_id] = node
        node_index_by_id[node_id] = i

    return id_to_node, node_index_by_id


def _evaluation_universe(net):
    """
    Universe là toàn bộ (frame, person) có layer trong Network.
    Mỗi layer có miss node, kể cả khi frame đó không có detection phù hợp.
    """
    units = set()

    for node in net.nodes:
        frame = _get(node, "frame")
        person = _get(node, "person")
        kind = _get(node, "kind")

        if frame is None or person is None:
            continue
        if kind not in ("det", "detection", "miss"):
            continue

        units.add((int(frame), str(person)))

    return sorted(units)


def _selected_body_id(node):
    if node is None:
        return None
    return _get(node, "body_id")


def _selected_pa_mm(node):
    if node is None:
        return np.nan
    value = _get(node, "pa_mm")
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _global_identity_metrics(gt_units, predicted_units):
    """
    IDF1 theo matching toàn cục giữa GT person và predicted WHAM body_id.

    gt_units:
      {(frame, person): True/False}

    predicted_units:
      {(frame, person): body_id hoặc None}

    Mỗi GT person-frame là một GT identity detection.
    Mỗi assignment detection là một predicted identity detection.
    """
    gt_count = sum(bool(v) for v in gt_units.values())

    predicted = [
        (frame, person, body_id)
        for (frame, person), body_id in predicted_units.items()
        if body_id is not None
    ]
    pred_count = len(predicted)

    gt_people = sorted(
        {person for (frame, person), present in gt_units.items() if present}
    )
    pred_ids = sorted(
        {body_id for _, _, body_id in predicted},
        key=lambda x: str(x),
    )

    if not gt_people or not pred_ids:
        idtp = 0
    else:
        gt_index = {p: i for i, p in enumerate(gt_people)}
        pred_index = {b: i for i, b in enumerate(pred_ids)}

        contingency = np.zeros(
            (len(gt_people), len(pred_ids)),
            dtype=np.int64,
        )

        for frame, person, body_id in predicted:
            if gt_units.get((frame, person), False):
                contingency[gt_index[person], pred_index[body_id]] += 1

        rows, cols = linear_sum_assignment(-contingency)
        idtp = int(contingency[rows, cols].sum())

    idfn = int(gt_count - idtp)
    idfp = int(pred_count - idtp)

    id_precision = _safe_div(idtp, idtp + idfp)
    id_recall = _safe_div(idtp, idtp + idfn)

    return {
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idf1": _f1(id_precision, id_recall),
    }


def evaluate_assignment(
    net,
    assignment: dict,
    truth: dict,
    method: str = "",
    sequence: str = "",
    solver_info: dict | None = None,
):
    """
    Đánh giá assignment của một phương pháp.

    assignment:
      {(frame, person): node_id hoặc None}

    truth:
      {(frame, body_id): person hoặc None}

    Accuracy:
      correct / (correct + wrong_id + missed)

    Detection P/R/F1:
      TP = correct
      FP = wrong_id
      FN = wrong_id + missed

    ID changes:
      số lần body_id thay đổi giữa hai assignment detection liên tiếp
      của cùng GT person; các frame miss được bỏ qua.
    """
    solver_info = solver_info or {}

    id_to_node, _ = _node_maps(net)
    units = _evaluation_universe(net)

    # truth có thể chứa person dạng int trong CSV, trong khi Network
    # dùng string hoặc ngược lại. Chuẩn hóa về string để so sánh.
    normalized_truth = {}
    for (frame, body_id), person in truth.items():
        normalized_truth[(int(frame), body_id)] = (
            None if person is None else str(person)
        )

    correct = 0
    wrong_id = 0
    missed = 0
    correct_miss = 0

    selected_pa = []
    predicted_units = {}
    gt_units = {}

    id_changes = 0
    previous_body = {}

    # Mỗi (frame, person) trong Network là một GT person-frame.
    for frame, person in units:
        node_id = assignment.get((frame, person))
        node = id_to_node.get(int(node_id)) if node_id is not None else None

        body_id = _selected_body_id(node)
        person_truth_ids = {
            b
            for (f, b), p in normalized_truth.items()
            if f == frame and p == person
        }

        gt_present = len(person_truth_ids) > 0
        gt_units[(frame, person)] = gt_present
        predicted_units[(frame, person)] = body_id

        if body_id is None:
            if gt_present:
                missed += 1
            else:
                correct_miss += 1
            continue

        pa_mm = _selected_pa_mm(node)
        if np.isfinite(pa_mm):
            selected_pa.append(pa_mm)

        if body_id in person_truth_ids:
            correct += 1
        else:
            wrong_id += 1

        # Đếm ID switch giữa các detection được chọn liên tiếp
        # sau khi bỏ qua những frame miss.
        if person in previous_body and previous_body[person] != body_id:
            id_changes += 1
        previous_body[person] = body_id

    evaluated = correct + wrong_id + missed
    accuracy = _safe_div(correct, evaluated)

    det_tp = correct
    det_fp = wrong_id
    det_fn = wrong_id + missed

    det_precision = _safe_div(det_tp, det_tp + det_fp)
    det_recall = _safe_div(det_tp, det_tp + det_fn)
    det_f1 = _f1(det_precision, det_recall)

    id_metrics = _global_identity_metrics(gt_units, predicted_units)

    result = {
        "method": method,
        "sequence": sequence,
        "n_units": len(units),
        "correct": correct,
        "correct_miss": correct_miss,
        "wrong_id": wrong_id,
        "missed": missed,
        "accuracy": accuracy,
        "id_changes": id_changes,
        "idtp": id_metrics["idtp"],
        "idfp": id_metrics["idfp"],
        "idfn": id_metrics["idfn"],
        "idf1": id_metrics["idf1"],
        "det_tp": det_tp,
        "det_fp": det_fp,
        "det_fn": det_fn,
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_f1": det_f1,
        "mean_pa_mpjpe_mm": (
            float(np.mean(selected_pa)) if selected_pa else np.nan
        ),
        "solver_method": solver_info.get("method", ""),
        "solver_status": solver_info.get(
            "status",
            solver_info.get("milp_status", ""),
        ),
        "solver_optimal": solver_info.get("optimal", np.nan),
        "solver_seconds": solver_info.get("seconds", np.nan),
        "objective": solver_info.get("objective", np.nan),
    }

    return result


def macro_average(results: pd.DataFrame):
    """
    Macro average: tính trung bình các metric theo sequence.
    Không cộng dồn count rồi mới chia, vì như vậy sẽ thành micro average.
    """
    if results.empty:
        return pd.DataFrame()

    numeric_metrics = [
        "accuracy",
        "idf1",
        "det_precision",
        "det_recall",
        "det_f1",
        "mean_pa_mpjpe_mm",
        "id_changes",
    ]

    rows = []

    for method, group in results.groupby("method", sort=False):
        row = {
            "method": method,
            "sequence": "MACRO_AVERAGE",
        }

        for metric in numeric_metrics:
            row[metric] = group[metric].mean()

        for count_metric in (
            "correct",
            "correct_miss",
            "wrong_id",
            "missed",
            "idtp",
            "idfp",
            "idfn",
            "det_tp",
            "det_fp",
            "det_fn",
        ):
            row[count_metric] = group[count_metric].sum()

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# build_network / tune: bảng gán, phân rã hàm mục tiêu, chấm điểm theo người
# ============================================================


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


def load_truth_by_person(path: Path) -> dict:
    """{(frame, người): {body_id,...}} từ mapping_frames.csv (bỏ dòng person rỗng)."""
    m = pd.read_csv(path, dtype={"person": str}).fillna({"person": ""})
    m = m[m["person"] != ""]
    truth: dict = {}
    for r in m.itertuples(index=False):
        truth.setdefault((int(r.frame), r.person), set()).add(int(r.body_id))
    return truth


def require_truth_by_person(truth: dict) -> dict:
    """Chặn nhầm định dạng. Nhãn của load_truth() tra theo (frame, người) luôn hụt,
    nên mọi detection bị chấm wrong_id mà không có lỗi nào."""
    for value in truth.values():
        if not isinstance(value, set):
            raise TypeError("Nhãn phải đọc bằng load_truth_by_person() "
                            "({(frame, người): {body_id}}), không phải load_truth().")
        break
    return truth


def status_of(body_id, truth_ids: set) -> str:
    if pd.isna(body_id):
        return "missed" if truth_ids else "correct_miss"
    return "correct" if int(body_id) in truth_ids else "wrong_id"


def evaluate(assign: pd.DataFrame, truth: dict) -> tuple[pd.DataFrame, dict]:
    require_truth_by_person(truth)
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