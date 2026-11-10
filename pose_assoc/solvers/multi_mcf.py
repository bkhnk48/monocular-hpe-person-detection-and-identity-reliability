"""
Min-cost flow nhiều luồng (mỗi người GT một luồng) có ràng buộc dùng chung detection.

  min  sum_a w_a x_a
  s.t. bảo toàn luồng tại mọi nút của từng người
       ra khỏi nguồn s_k = 1,  vào đích t_k = 1
       sum_k (luồng vào nút (t, d, k)) <= 1   với mọi detection (t, d)
       x_a in {0, 1}

Giải bằng scipy.optimize.milp (HiGHS). Trước đó thử giải độc lập từng người bằng
quy hoạch động: nếu không detection nào bị 2 người cùng dùng thì nghiệm đó đã
tối ưu (bài toán nới lỏng có cùng nghiệm), khỏi phải gọi MILP.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..network import Network
from .single_mcf import shortest_paths


def shared_detections(net: Network, flow: np.ndarray) -> int:
    """Số detection (frame, body_id) bị nhiều hơn 1 người dùng."""
    used: dict = {}
    for a in np.flatnonzero(flow):
        n = net.nodes[net.heads[a]]
        if n["kind"] == "det":
            key = (n["frame"], n["body_id"])
            used[key] = used.get(key, 0) + 1
    return sum(1 for c in used.values() if c > 1)


def conservation(net: Network) -> LinearConstraint:
    n_arc = len(net.tails)
    cols = np.arange(n_arc)
    A = coo_matrix((np.r_[np.ones(n_arc), -np.ones(n_arc)],
                    (np.r_[net.heads, net.tails], np.r_[cols, cols])),
                   shape=(len(net.nodes), n_arc)).tocsr()
    b = np.array([{"source": -1.0, "sink": 1.0}.get(n["kind"], 0.0) for n in net.nodes])
    return LinearConstraint(A, b, b)


def coupling(net: Network) -> LinearConstraint | None:
    keys: dict = {}
    rows, cols = [], []
    for a, h in enumerate(net.heads):
        n = net.nodes[h]
        if n["kind"] == "det":
            rows.append(keys.setdefault((n["frame"], n["body_id"]), len(keys)))
            cols.append(a)
    if not keys:
        return None
    A = coo_matrix((np.ones(len(rows)), (rows, cols)),
                   shape=(len(keys), len(net.tails))).tocsr()
    return LinearConstraint(A, -np.inf, 1.0)


# Mã trạng thái của scipy.optimize.milp
MILP_STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible",
               3: "unbounded", 4: "other"}


def solve_milp(net: Network, time_limit: float) -> tuple[np.ndarray, dict]:
    """Trả về (luồng, trạng thái HiGHS).

    status = 0 mới là tối ưu đã chứng minh. status = 1 (hết giờ) mà vẫn có res.x
    thì đó chỉ là nghiệm khả thi tốt nhất tìm được; mip_gap cho biết còn cách
    cận dưới bao xa.
    """
    cons = [c for c in (conservation(net), coupling(net)) if c is not None]
    res = milp(c=net.arc_weight(), constraints=cons,
               integrality=np.ones(len(net.tails)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    stats = {
        "success": bool(res.success),
        "status": int(res.status),
        "status_name": MILP_STATUS.get(int(res.status), "unknown"),
        "message": str(res.message),
        "mip_gap": _num(getattr(res, "mip_gap", None)),
        "mip_node_count": _num(getattr(res, "mip_node_count", None)),
        "mip_dual_bound": _num(getattr(res, "mip_dual_bound", None)),
    }
    if res.x is None:
        raise RuntimeError(f"MILP không tìm được nghiệm: {res.message} (status {res.status})")
    return np.round(res.x).astype(int), stats


def _num(x):
    return None if x is None else float(x)


def solve_network(net: Network, time_limit: float = 300.0) -> tuple[np.ndarray, dict]:
    """Trả về (luồng trên từng cung, thông tin bộ giải)."""
    t0 = time.time()
    flow = shortest_paths(net)
    shared = shared_detections(net, flow)
    info = {"K": len(net.persons), "dp_shared_detections": shared}
    if len(net.persons) <= 1 or shared == 0:
        info["method"] = "dp (tối ưu chính xác)"
        info["optimal"] = True
    else:
        flow, stats = solve_milp(net, time_limit)
        info["method"] = "milp (HiGHS)"
        info["optimal"] = stats["status"] == 0
        info["milp"] = stats
        if not info["optimal"]:
            warnings.warn(f"MILP chưa chứng minh tối ưu ({stats['status_name']}, "
                          f"mip_gap={stats['mip_gap']}): nghiệm chỉ là khả thi.")
    info["seconds"] = round(time.time() - t0, 2)
    info["objective"] = float(net.arc_weight() @ flow + net.constant)
    return flow, info
