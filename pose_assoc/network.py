"""
Dựng mạng time-expanded cho bài toán gán detection WHAM với người GT.

Mỗi người GT k là một luồng (commodity) mang 1 đơn vị, đi từ nguồn s_k tới đích t_k.
Với mỗi frame t mà người k có trong GT, có một lớp gồm:
  - nút (t, miss, k)          : người k không được gán detection nào (FN)
  - nút (t, d, k) cho mỗi d   : detection d (body_id) được gán cho người k
Cung nối mọi nút của lớp này với mọi nút của lớp kế tiếp (cùng người k).

Ràng buộc nối các luồng: mỗi detection (t, d) chỉ thuộc tối đa 1 người.
Số luồng K = số người trong GT.

Mạng lưu ĐẠI LƯỢNG THÔ (PA-MPJPE của nút, rms chuyển động của cung, loại cung,
độ dịch 2D của cung nếu đã gọi attach_positions(), sai số 2D so với GT của nút
detection nếu đã gọi attach_gt_2d()). C_ID (đổi body_id) tính thẳng từ body_id của nút.
Chi phí là hàm tuyến tính của các hệ số, nên đổi hệ số chỉ cần gọi apply_params()
mà không phải dựng lại mạng (dùng cho pose_assoc.tune).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import (CostParams, detection_cost, displacement_2d, displacement_cost,
                    id_switch_cost, motion_cost, motion_rms, procrustes, reprojection_cost,
                    reprojection_error_2d)

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
    # độ dịch 2D (C_V), chỉ khác 0 ở cung "dd"; rỗng = chưa gọi attach_positions()
    arc_disp: list[float] = field(default_factory=list)
    # True sau attach_gt_2d(): nút detection có "e2d" (C_2D); mạng cũ trong cache không có
    has_gt_2d: bool = False
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
    # C_2D: chỉ có khi đã gắn attach_gt_2d(); chưa gắn (kể cả lúc build_network tự gọi
    # apply_params ở cuối) thì bỏ qua như C_V. tune chặn quét beta trên cache chưa gắn.
    if p.beta and has_gt_2d(net):
        det_cost = det_cost + reprojection_cost(node_e2d(net), p)
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
    # C_V: chỉ có khi đã gắn attach_positions(); mạng chưa gắn (kể cả lúc build_network
    # tự gọi apply_params ở cuối) thì bỏ qua. Cache cũ đã bị tune.load_cached() chặn.
    disp = np.asarray(arc_disp(net), float)
    if p.eta and len(disp):
        if len(disp) != len(sub):
            raise ValueError(f"arc_disp có {len(disp)} phần tử nhưng mạng có {len(sub)} cung")
        net.arc_cost[dd] += displacement_cost(disp[dd], p, gap[dd])
    # C_ID: tính từ body_id có sẵn trên nút, nên mạng cũ trong cache cũng dùng được
    if p.mu:
        net.arc_cost += id_switch_cost(arc_id_switch(net), p)
    net.constant = p.c_fp * net.n_detections
    return net


def arc_id_switch(net: Network) -> np.ndarray:
    """True ở cung Detected -> Detected mà hai detection khác body_id (C_ID).

    Cung qua trạng thái missed (dm, md) không mang body_id nên không bị tính: đi
    det(body A) -> miss -> det(body B) không bị phạt C_ID.
    """
    body = np.array([n["body_id"] for n in net.nodes])
    tails, heads = np.asarray(net.tails, int), np.asarray(net.heads, int)
    return (np.asarray(net.arc_sub) == "dd") & (body[tails] != body[heads])


def has_gt_2d(net: Network) -> bool:
    """Đã gắn sai số 2D so với GT (attach_gt_2d) chưa; mạng cũ trong cache thì chưa."""
    return bool(net.__dict__.get("has_gt_2d", False))


def node_e2d(net: Network) -> np.ndarray:
    """E_2D của mọi nút (NaN với nút không phải detection hoặc thiếu GT 2D)."""
    return np.array([n.get("e2d", np.nan) for n in net.nodes], float)


def attach_gt_2d(net: Network, det_uv: dict, gt_uv: dict) -> Network:
    """Gắn sai số 2D thô E_2D cho mọi nút detection (C_2D).

    det_uv: {(frame, body_id): (12, 2) px} (WHAM-2D.csv);
    gt_uv:  {(frame, người): (12, 2) px} (GT 2D trên ảnh của cùng video), thứ tự khớp LIMBS.
    Gọi apply_params() lại sau đó thì C_2D mới vào chi phí. Báo lỗi nếu không nút
    detection nào tính được E_2D (thường do tên người hoặc số frame của GT 2D không khớp).
    """
    n_det = n_ok = 0
    for n in net.nodes:
        if n["kind"] != "det":
            continue
        n["e2d"] = reprojection_error_2d(det_uv.get((n["frame"], n["body_id"])),
                                         gt_uv.get((n["frame"], n["person"])))
        n_det += 1
        n_ok += bool(np.isfinite(n["e2d"]))
    if n_det and not n_ok:
        raise ValueError("GT 2D không khớp nút detection nào: kiểm tra tên người "
                         f"({', '.join(net.persons)}) và số frame của GT 2D")
    net.has_gt_2d = True
    net.gt_2d_coverage = (n_ok, n_det)
    return net


def arc_disp(net: Network) -> list:
    """Độ dịch 2D của các cung; [] nếu chưa gắn (mạng cũ trong cache không có trường này)."""
    disp = net.__dict__.get("arc_disp")
    return [] if disp is None else disp


def attach_positions(net: Network, det_uv: dict) -> Network:
    """Gắn độ dịch 2D thô cho mọi cung (0 nếu không phải cung "dd").

    det_uv: {(frame, body_id): (12, 2) px}, cùng thứ tự khớp LIMBS. Chỉ lưu con số
    vào net.arc_disp, không lưu toạ độ vào nút, để cache của tune nhẹ.
    Gọi apply_params() lại sau đó thì C_V mới vào chi phí.
    """
    def uv(i):
        n = net.nodes[i]
        return det_uv.get((n["frame"], n["body_id"]))

    net.arc_disp = [displacement_2d(uv(a), uv(b)) if s == "dd" else 0.0
                    for a, b, s in zip(net.tails, net.heads, net.arc_sub)]
    return net


def build_layer(net: Network, frame: int, person: str, gt_pose, dets) -> list[int]:
    """Các nút của người `person` ở `frame`: 1 nút miss + các detection căn được."""
    ids = [net.add_node(kind="miss", person=person, frame=frame, body_id=-1,
                        pa_mm=np.nan, residual=None, e2d=np.nan)]
    for body_id, pose in dets:
        pa, res = procrustes(pose, gt_pose)
        if np.isfinite(pa):
            ids.append(net.add_node(kind="det", person=person, frame=frame,
                                    body_id=int(body_id), pa_mm=pa * 1000.0, residual=res,
                                    e2d=np.nan))
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
    common = dict(person=person, frame=-1, body_id=-1, pa_mm=np.nan, residual=None,
                  e2d=np.nan)
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
    if has_gt_2d(net):
        df["e2d"] = node_e2d(net)
    df["cost"] = net.node_costs
    return df


def arcs_frame(net: Network, flow: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"tail": net.tails, "head": net.heads, "kind": net.arc_kind,
                       "sub": net.arc_sub, "rms_m": net.arc_rms,
                       "gap": net.__dict__.get("arc_gap") or 1,
                       "arc_cost": net.arc_cost, "weight": net.arc_weight()})
    df["person"] = [net.nodes[t]["person"] for t in net.tails]
    if len(arc_disp(net)) == len(df):
        df["disp_2d"] = arc_disp(net)
    df["id_switch"] = arc_id_switch(net)
    df.insert(0, "id", range(len(df)))
    if flow is not None:
        df["flow"] = flow.astype(int)
    return df
