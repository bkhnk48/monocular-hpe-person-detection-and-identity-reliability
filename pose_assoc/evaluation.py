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

Chấm theo mức che (pose_assoc.compare, strata.csv): mỗi (frame, người GT) thuộc một tầng
theo MaxIoU giữa khung 2D GT của người đó và khung của người GT khác cùng frame
(occlusion_strata); stratified_rows() đếm lại các trạng thái trên trong từng tầng.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .costs import (MIN_VALID_JOINTS, CostParams, displacement_cost, id_switch_cost,
                    reprojection_cost)
from .network import Network, arc_disp, arc_id_switch, has_gt_2d


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


def _truth_index(truth: dict) -> dict:
    """{(frame, người): {body_id}} từ nhãn {(frame, body_id): người hoặc None}.
    Người được chuẩn hoá về chuỗi (CSV có thể đọc ra số)."""
    index: dict = {}
    for (frame, body_id), person in truth.items():
        if person is not None:
            index.setdefault((int(frame), str(person)), set()).add(body_id)
    return index


def unit_outcomes(net, assignment: dict, truth: dict) -> list[dict]:
    """Kết quả của từng (frame, người GT) trong mạng, theo thứ tự frame.

    status: correct / wrong_id / missed / correct_miss (như evaluate());
    id_change: body_id được chọn khác detection được chọn liền trước của cùng người
    (các frame missed bỏ qua). evaluate_assignment() và stratified_rows() cùng đếm từ đây.
    """
    id_to_node, _ = _node_maps(net)
    index = _truth_index(truth)
    previous, rows = {}, []
    for frame, person in _evaluation_universe(net):
        node_id = assignment.get((frame, person))
        node = id_to_node.get(int(node_id)) if node_id is not None else None
        body_id = _selected_body_id(node)
        ids = index.get((frame, person), set())
        row = {"frame": frame, "person": person, "body_id": body_id,
               "gt_present": bool(ids), "pa_mm": np.nan, "id_change": False}
        if body_id is None:
            row["status"] = "missed" if ids else "correct_miss"
        else:
            row["pa_mm"] = _selected_pa_mm(node)
            row["status"] = "correct" if body_id in ids else "wrong_id"
            row["id_change"] = person in previous and previous[person] != body_id
            previous[person] = body_id
        rows.append(row)
    return rows


# Ngưỡng "che nặng" theo OCHuman (Zhang et al., CVPR 2019): MaxIoU > 0.5.
HEAVY_IOU = 0.5
STRATA = ("tách", "chồng", "chồng nặng", "không rõ")


def joint_box(uv) -> tuple | None:
    """Khung bao các khớp 2D hợp lệ (u0, v0, u1, v1); None nếu ít hơn 8 khớp."""
    uv = np.asarray(uv, float)
    ok = np.isfinite(uv).all(1)
    if ok.sum() < MIN_VALID_JOINTS:
        return None
    pts = uv[ok]
    return (*pts.min(0), *pts.max(0))


def box_iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def stratum_of(max_iou: float, heavy_iou: float = HEAVY_IOU) -> str:
    if not np.isfinite(max_iou):
        return "không rõ"
    if max_iou <= 0.0:
        return "tách"
    return "chồng nặng" if max_iou > heavy_iou else "chồng"


def occlusion_strata(net, gt_uv: dict, heavy_iou: float = HEAVY_IOU) -> dict:
    """Mức che của từng (frame, người GT) trong mạng, chỉ từ GT (không phụ thuộc phương pháp).

    gt_uv: {(frame, người): (12, 2) px}. MaxIoU = IoU lớn nhất giữa khung bao 12 khớp GT 2D
    của người đó và của người GT khác có mặt cùng frame (người duy nhất trong frame: 0).
    Thiếu GT 2D của người đó hoặc của người kia thì MaxIoU = NaN (tầng "không rõ").
    Trả về {(frame, người): (MaxIoU, tầng)}.
    """
    people_at: dict = defaultdict(list)
    for frame, person in _evaluation_universe(net):
        people_at[frame].append(person)
    out = {}
    for frame, people in people_at.items():
        boxes = {p: joint_box(gt_uv[(frame, p)]) if (frame, p) in gt_uv else None
                 for p in people}
        for p in people:
            others = [q for q in people if q != p]
            if boxes[p] is None or any(boxes[q] is None for q in others):
                m = float("nan")
            else:
                m = max((box_iou(boxes[p], boxes[q]) for q in others), default=0.0)
            out[(frame, p)] = (m, stratum_of(m, heavy_iou))
    return out


def stratified_rows(net, assignment: dict, truth: dict, strata: dict,
                    method: str = "", sequence: str = "") -> list[dict]:
    """Đếm trạng thái của một phương pháp trong từng tầng che (cùng định nghĩa accuracy
    với evaluate_assignment: correct / (correct + wrong_id + missed))."""
    counts: dict = {}
    for r in unit_outcomes(net, assignment, truth):
        _, stratum = strata.get((r["frame"], r["person"]), (np.nan, "không rõ"))
        c = counts.setdefault(stratum, {"n_units": 0, "correct": 0, "wrong_id": 0,
                                        "missed": 0, "correct_miss": 0, "id_changes": 0})
        c["n_units"] += 1
        c[r["status"]] += 1
        c["id_changes"] += int(r["id_change"])
    rows = []
    for stratum in STRATA:
        if stratum in counts:
            c = counts[stratum]
            rows.append({"method": method, "sequence": sequence, "stratum": stratum, **c,
                         "accuracy": _safe_div(c["correct"],
                                               c["correct"] + c["wrong_id"] + c["missed"])})
    return rows


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

    rows = unit_outcomes(net, assignment, truth)
    status = [r["status"] for r in rows]
    correct, wrong_id = status.count("correct"), status.count("wrong_id")
    missed, correct_miss = status.count("missed"), status.count("correct_miss")
    selected_pa = [r["pa_mm"] for r in rows
                   if r["body_id"] is not None and np.isfinite(r["pa_mm"])]
    id_changes = sum(r["id_change"] for r in rows)
    gt_units = {(r["frame"], r["person"]): r["gt_present"] for r in rows}
    predicted_units = {(r["frame"], r["person"]): r["body_id"] for r in rows}

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
        "n_units": len(rows),
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
    # C_2D nằm trong chi phí nút detection, tách riêng khỏi pose (PA-MPJPE)
    reprojection = 0.0
    if p.beta and has_gt_2d(net):
        e2d = np.array([n.get("e2d", np.nan) for n in dets], float)
        reprojection = float(reprojection_cost(e2d, p).sum())
    n_fn = sum(n["kind"] == "miss" for n in used)
    # C_V nằm sẵn trong arc_cost của cung "dd": tách ra để không đếm hai lần
    disp = np.asarray(arc_disp(net), float)
    displacement = 0.0
    if p.eta and len(disp) == len(net.arc_sub):
        on = (np.asarray(flow) == 1) & (np.asarray(net.arc_sub) == "dd")
        gap = np.asarray(net.__dict__.get("arc_gap") or np.ones(len(on)), float)
        displacement = float(displacement_cost(disp[on], p, gap[on]).sum())
    # C_ID cũng nằm trong arc_cost của cung "dd": tách ra như C_V
    switches = arc_id_switch(net)[np.asarray(flow) == 1]
    id_switch = float(id_switch_cost(switches, p).sum()) if p.mu else 0.0
    return {"pose": pose,
            "reprojection": reprojection,
            "false_positive": p.c_fp * (n_det_total - len(dets)),
            "false_negative": p.c_fn * n_fn,
            "temporal_and_gaps": (float(arc_costs[kinds == "move"].sum())
                                  - displacement - id_switch),
            "displacement": displacement,
            "id_switch": id_switch,
            "n_id_switches": int(switches.sum()),
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