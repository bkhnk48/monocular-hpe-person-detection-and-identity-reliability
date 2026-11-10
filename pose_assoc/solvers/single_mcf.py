"""
Một luồng / mỗi người: đường đi ngắn nhất trên DAG phân lớp (quy hoạch động).

Với K = 1 (MPI-INF-3DHP) đây là nghiệm tối ưu chính xác của min-cost flow,
tương đương scripts/solve_mc_mcf.py.
"""

from __future__ import annotations

import numpy as np

from ..network import Network


def shortest_paths(net: Network) -> np.ndarray:
    """Giải độc lập từng người (bỏ qua ràng buộc dùng chung detection).

    Cung được thêm theo đúng thứ tự topo (start -> các lớp -> end),
    nên chỉ cần duyệt cung theo thứ tự chỉ số.
    """
    w = net.arc_weight()
    tails, heads = np.asarray(net.tails), np.asarray(net.heads)
    dist = np.full(len(net.nodes), np.inf)
    pred = np.full(len(net.nodes), -1)
    for n in net.nodes:
        if n["kind"] == "source":
            dist[n["id"]] = 0.0
    for a in range(len(w)):
        cand = dist[tails[a]] + w[a]
        if cand < dist[heads[a]]:
            dist[heads[a]], pred[heads[a]] = cand, a
    return backtrack(net, pred)


def backtrack(net: Network, pred: np.ndarray) -> np.ndarray:
    flow = np.zeros(len(net.tails), int)
    for n in net.nodes:
        if n["kind"] != "sink":
            continue
        v = n["id"]
        while pred[v] >= 0:
            flow[pred[v]] = 1
            v = net.tails[pred[v]]
    return flow
