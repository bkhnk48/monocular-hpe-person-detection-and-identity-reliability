"""
Dựng mạng time-expanded cho bài toán gán detection WHAM với người GT.

Mỗi người GT k là một luồng (commodity) mang 1 đơn vị, đi từ nguồn s_k tới đích t_k.
Với mỗi frame t mà người k có trong GT, có một lớp gồm:
  - nút (t, miss, k)          : người k không được gán detection nào (FN)
  - nút (t, d, k) cho mỗi d   : detection d (body_id) được gán cho người k
Cung nối mọi nút của lớp này với mọi nút của lớp kế tiếp (cùng người k).

Ràng buộc nối các luồng: mỗi detection (t, d) chỉ thuộc tối đa 1 người.
Số luồng K = số người trong GT.

Mạng lưu ĐẠI LƯỢNG THÔ (PA-MPJPE của nút, rms chuyển động của cung, loại cung).
Chi phí là hàm tuyến tính của các hệ số, nên đổi hệ số chỉ cần gọi apply_params()
mà không phải dựng lại mạng (dùng cho pose_assoc.tune).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostParams, detection_cost, motion_cost, motion_rms, procrustes

# Loại cung chi tiết (arc_sub) -> loại chung (arc_kind)
SUB_TO_KIND = {"dd": "move", "dm": "move", "md": "move", "mm": "move",
               "start_det": "start", "start_miss": "start",
               "end_det": "end", "end_miss": "end", "empty": "empty"}


@dataclass
class Network:
    persons: list[str]
    nodes: list[dict] = field(default_factory=list)
    tails: list[int] = field(default_factory=list)
    heads: list[int] = field(default_factory=list)
    arc_sub: list[str] = field(default_factory=list)
    arc_rms: list[float] = field(default_factory=list)   # chỉ có nghĩa với cung "dd"
    arc_gap: list[int] = field(default_factory=list)     # dt (frame gốc) của cung move
    arc_cost: np.ndarray = None
    node_costs: np.ndarray = None
    n_detections: int = 0
    constant: float = 0.0            # C_FP * tổng số detection

    @property
    def arc_kind(self) -> list[str]:
        return [SUB_TO_KIND[s] for s in self.arc_sub]

    def add_node(self, **kw) -> int:
        kw["id"] = len(self.nodes)
        self.nodes.append(kw)
        return kw["id"]

    def add_arc(self, tail: int, head: int, sub: str, rms: float = 0.0, gap: int = 1):
        self.tails.append(tail)
        self.heads.append(head)
        self.arc_sub.append(sub)
        self.arc_rms.append(float(rms))
        self.arc_gap.append(int(gap))

    def node_cost(self) -> np.ndarray:
        return self.node_costs

    def arc_weight(self) -> np.ndarray:
        """Trọng số cung dùng khi giải = chi phí cung + chi phí nút đầu cuối (head)."""
        return self.arc_cost + self.node_costs[np.asarray(self.heads)]


def apply_params(net: Network, p: CostParams) -> Network:
    """Tính chi phí nút và cung từ đại lượng thô theo bộ hệ số p."""
    kind = np.array([n["kind"] for n in net.nodes])
    pa = np.array([n["pa_mm"] for n in net.nodes], float)
    det_cost = np.array([detection_cost(x / 1000.0, p) for x in np.nan_to_num(pa)])
    net.node_costs = np.where(kind == "det", det_cost, np.where(kind == "miss", p.c_fn, 0.0))
    fixed = {"dm": p.c_disappear, "md": p.c_reappear, "mm": 0.0, "empty": 0.0,
             "start_det": p.start_detected, "start_miss": p.start_missed,
             "end_det": p.end_detected, "end_miss": p.end_missed, "dd": 0.0}
    sub = np.array(net.arc_sub)
    net.arc_cost = np.array([fixed[s] for s in net.arc_sub], float)
    dd = sub == "dd"
    # mạng cũ trong cache tuning chưa có arc_gap -> coi mọi dt = 1
    gap = np.asarray(net.__dict__.get("arc_gap") or np.ones(len(sub)), float)
    net.arc_cost[dd] = motion_cost(np.asarray(net.arc_rms)[dd], p, gap[dd])
    net.constant = p.c_fp * net.n_detections
    return net


def build_layer(net: Network, frame: int, person: str, gt_pose, dets) -> list[int]:
    """Các nút của người `person` ở `frame`: 1 nút miss + các detection căn được."""
    ids = [net.add_node(kind="miss", person=person, frame=frame, body_id=-1,
                        pa_mm=np.nan, residual=None)]
    for body_id, pose in dets:
        pa, res = procrustes(pose, gt_pose)
        if np.isfinite(pa):
            ids.append(net.add_node(kind="det", person=person, frame=frame,
                                    body_id=int(body_id), pa_mm=pa * 1000.0, residual=res))
    return ids


def connect(net: Network, prev: list[int], cur: list[int]):
    gap = net.nodes[cur[0]]["frame"] - net.nodes[prev[0]]["frame"]
    for a in prev:
        for b in cur:
            na, nb = net.nodes[a], net.nodes[b]
            sub = ("d" if na["kind"] == "det" else "m") + ("d" if nb["kind"] == "det" else "m")
            rms = motion_rms(na["residual"], nb["residual"]) if sub == "dd" else 0.0
            net.add_arc(a, b, sub, rms, gap)


def build_person(net: Network, person: str, gt: dict, dets: dict):
    frames = sorted(f for f, people in gt.items() if person in people)
    common = dict(person=person, frame=-1, body_id=-1, pa_mm=np.nan, residual=None)
    src = net.add_node(kind="source", **common)
    sink = net.add_node(kind="sink", **common)
    prev = None
    for f in frames:
        cur = build_layer(net, f, person, gt[f][person], dets.get(f, []))
        if prev is None:
            for b in cur:
                det = net.nodes[b]["kind"] == "det"
                net.add_arc(src, b, "start_det" if det else "start_miss")
        else:
            connect(net, prev, cur)
        prev = cur
    for a in prev or []:
        det = net.nodes[a]["kind"] == "det"
        net.add_arc(a, sink, "end_det" if det else "end_miss")
    if prev is None:
        net.add_arc(src, sink, "empty")


def build_network(dets: dict, gt: dict, p: CostParams) -> Network:
    """dets: {frame: [(body_id, pose12)]}; gt: {frame: {người: pose12}}."""
    persons = sorted({k for people in gt.values() for k in people})
    net = Network(persons=persons)
    for person in persons:
        build_person(net, person, gt, dets)
    for n in net.nodes:          # residual chỉ cần lúc dựng; bỏ đi cho nhẹ khi lưu
        n["residual"] = None
    net.n_detections = sum(len(v) for v in dets.values())
    return apply_params(net, p)


def nodes_frame(net: Network) -> pd.DataFrame:
    cols = ["id", "kind", "person", "frame", "body_id", "pa_mm"]
    df = pd.DataFrame([{c: n[c] for c in cols} for n in net.nodes])
    df["cost"] = net.node_costs
    return df


def arcs_frame(net: Network, flow: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"tail": net.tails, "head": net.heads, "kind": net.arc_kind,
                       "sub": net.arc_sub, "rms_m": net.arc_rms,
                       "gap": net.__dict__.get("arc_gap") or 1,
                       "arc_cost": net.arc_cost, "weight": net.arc_weight()})
    df["person"] = [net.nodes[t]["person"] for t in net.tails]
    df.insert(0, "id", range(len(df)))
    if flow is not None:
        df["flow"] = flow.astype(int)
    return df
