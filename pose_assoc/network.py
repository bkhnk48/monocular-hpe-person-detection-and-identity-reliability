"""
Dựng mạng time-expanded cho bài toán gán detection WHAM với người GT.

Mỗi người GT k là một luồng (commodity) mang 1 đơn vị, đi từ nguồn s_k tới đích t_k.
Với mỗi frame t mà người k có trong GT, có một lớp gồm:
  - nút (t, miss, k)          : người k không được gán detection nào (FN)
  - nút (t, d, k) cho mỗi d   : detection d (body_id) được gán cho người k
Cung nối mọi nút của lớp này với mọi nút của lớp kế tiếp (cùng người k).

Ràng buộc nối các luồng: mỗi detection (t, d) chỉ thuộc tối đa 1 người.
Số luồng K = số người trong GT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostParams, detection_cost, procrustes, transition_cost


@dataclass
class Network:
    persons: list[str]
    nodes: list[dict] = field(default_factory=list)
    tails: list[int] = field(default_factory=list)
    heads: list[int] = field(default_factory=list)
    arc_cost: list[float] = field(default_factory=list)
    arc_kind: list[str] = field(default_factory=list)
    constant: float = 0.0            # C_FP * tổng số detection

    def add_node(self, **kw) -> int:
        kw["id"] = len(self.nodes)
        self.nodes.append(kw)
        return kw["id"]

    def add_arc(self, tail: int, head: int, cost: float, kind: str):
        self.tails.append(tail)
        self.heads.append(head)
        self.arc_cost.append(float(cost))
        self.arc_kind.append(kind)

    def node_cost(self) -> np.ndarray:
        return np.array([n["cost"] for n in self.nodes], float)

    def arc_weight(self) -> np.ndarray:
        """Trọng số cung dùng khi giải = chi phí cung + chi phí nút đầu cuối (head)."""
        return np.asarray(self.arc_cost) + self.node_cost()[np.asarray(self.heads)]


def build_layer(net: Network, frame: int, person: str, gt_pose, dets, p: CostParams):
    """Các nút của người `person` ở `frame`: 1 nút miss + các detection căn được."""
    ids = [net.add_node(kind="miss", person=person, frame=frame, body_id=-1,
                        pa_mm=np.nan, cost=p.c_fn, residual=None)]
    for body_id, pose in dets:
        pa, res = procrustes(pose, gt_pose)
        if np.isfinite(pa):
            ids.append(net.add_node(kind="det", person=person, frame=frame,
                                    body_id=int(body_id), pa_mm=pa * 1000.0,
                                    cost=detection_cost(pa, p), residual=res))
    return ids


def connect(net: Network, prev: list[int], cur: list[int], p: CostParams):
    for a in prev:
        for b in cur:
            net.add_arc(a, b, transition_cost(net.nodes[a], net.nodes[b], p), "move")


def build_person(net: Network, person: str, gt: dict, dets: dict, p: CostParams):
    frames = sorted(f for f, people in gt.items() if person in people)
    src = net.add_node(kind="source", person=person, frame=-1, body_id=-1,
                       pa_mm=np.nan, cost=0.0, residual=None)
    sink = net.add_node(kind="sink", person=person, frame=-1, body_id=-1,
                        pa_mm=np.nan, cost=0.0, residual=None)
    prev = None
    for f in frames:
        cur = build_layer(net, f, person, gt[f][person], dets.get(f, []), p)
        if prev is None:
            for b in cur:
                det = net.nodes[b]["kind"] == "det"
                net.add_arc(src, b, p.start_detected if det else p.start_missed, "start")
        else:
            connect(net, prev, cur, p)
        prev = cur
    for a in prev or []:
        det = net.nodes[a]["kind"] == "det"
        net.add_arc(a, sink, p.end_detected if det else p.end_missed, "end")
    if prev is None:
        net.add_arc(src, sink, 0.0, "empty")


def build_network(dets: dict, gt: dict, p: CostParams) -> Network:
    """dets: {frame: [(body_id, pose12)]}; gt: {frame: {người: pose12}}."""
    persons = sorted({k for people in gt.values() for k in people})
    net = Network(persons=persons)
    for person in persons:
        build_person(net, person, gt, dets, p)
    net.constant = p.c_fp * sum(len(v) for v in dets.values())
    return net


def nodes_frame(net: Network) -> pd.DataFrame:
    cols = ["id", "kind", "person", "frame", "body_id", "pa_mm", "cost"]
    return pd.DataFrame([{c: n[c] for c in cols} for n in net.nodes])


def arcs_frame(net: Network, flow: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"tail": net.tails, "head": net.heads, "kind": net.arc_kind,
                       "arc_cost": net.arc_cost, "weight": net.arc_weight()})
    df["person"] = [net.nodes[t]["person"] for t in net.tails]
    df.insert(0, "id", range(len(df)))
    if flow is not None:
        df["flow"] = flow.astype(int)
    return df
