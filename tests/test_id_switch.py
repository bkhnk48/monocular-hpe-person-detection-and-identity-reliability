"""C_ID (mu): phạt khi một người GT đi từ body_id này sang body_id khác giữa hai detection
liên tiếp (dùng ID tracker của WHAM, như B5 của pose_assoc.compare).

Chạy:  python3 -m pytest -q tests/test_id_switch.py
"""
import numpy as np
import pytest

from pose_assoc import tune
from pose_assoc.baselines import hungarian_per_frame, independent_dp
from pose_assoc.build_network import OVERRIDES, load_params
from pose_assoc.costs import CostParams, id_switch_cost
from pose_assoc.evaluation import assignment_table, objective_breakdown
from pose_assoc.network import apply_params, arc_id_switch, arcs_frame, build_network
from pose_assoc.solvers.multi_mcf import solve_network

RNG = np.random.default_rng(0)
POSE = RNG.normal(scale=0.3, size=(12, 3))            # pose GT 3D (m)
OTHER = RNG.normal(scale=0.3, size=(12, 3))           # pose của một người khác


def switch_scene(frames=range(5)):
    """1 người GT. body 0 = đúng người (PA nhỏ). body 1 = người khác, riêng frame 2 có
    pose trùng hẳn GT (PA = 0), nên chỉ xét pose thì đường đi nhảy sang body 1 ở frame 2
    rồi quay về body 0: hai lần đổi body_id."""
    rng = np.random.default_rng(1)
    frames = list(frames)
    gt = {f: {"P": POSE} for f in frames}
    dets = {f: [(0, POSE + rng.normal(scale=0.01, size=(12, 3))),
                (1, POSE.copy() if f == 2 else OTHER)] for f in frames}
    return dets, gt


def chosen(net, p):
    apply_params(net, p)
    flow, info = solve_network(net)
    rows = assignment_table(net, flow).sort_values("frame")
    return [int(b) for b in rows["body_id"].fillna(-1)], flow, info   # -1 = missed


def test_formula_is_mu_times_indicator():
    p = CostParams(mu=1.5)
    assert id_switch_cost([True, False, True], p) == pytest.approx([1.5, 0.0, 1.5])
    assert id_switch_cost([True], CostParams()) == pytest.approx([0.0])     # mu = 0


def test_mu_keeps_identity_that_pose_alone_switches():
    dets, gt = switch_scene()
    p = CostParams(gamma=0.0)
    net = build_network(dets, gt, p)
    assert chosen(net, p)[0] == [0, 0, 1, 0, 0]
    # đổi qua lại tốn 2 * mu: mu nhỏ hơn nửa phần lợi về pose thì vẫn đổi
    assert chosen(net, CostParams(gamma=0.0, mu=0.001))[0] == [0, 0, 1, 0, 0]
    assert chosen(net, CostParams(gamma=0.0, mu=1.0))[0] == [0, 0, 0, 0, 0]


def test_penalty_only_on_detected_to_detected_arcs_that_change_body():
    dets, gt = switch_scene()
    net = build_network(dets, gt, CostParams())
    base = apply_params(net, CostParams()).arc_cost.copy()
    cost = apply_params(net, CostParams(mu=0.7)).arc_cost
    switch = arc_id_switch(net)
    assert switch.sum() == 4 * 2                      # 4 cặp frame, mỗi cặp 0->1 và 1->0
    for a, (sub, sw) in enumerate(zip(net.arc_sub, switch)):
        body = (net.nodes[net.tails[a]]["body_id"], net.nodes[net.heads[a]]["body_id"])
        assert sw == (sub == "dd" and body[0] != body[1])
        assert cost[a] - base[a] == pytest.approx(0.7 if sw else 0.0)
    assert np.array_equal(net.node_costs, apply_params(net, CostParams()).node_costs)
    assert arcs_frame(net)["id_switch"].sum() == 8


def test_penalty_not_divided_by_gap():
    dets, gt = switch_scene(frames=[0, 3, 4])          # GT thiếu frame 1, 2 -> dt = 3
    net = build_network(dets, gt, CostParams())
    base = apply_params(net, CostParams(gamma=0.0)).arc_cost.copy()
    cost = apply_params(net, CostParams(gamma=0.0, mu=0.5)).arc_cost
    gap3 = np.asarray(net.arc_gap) == 3
    assert (arc_id_switch(net) & gap3).any()
    assert cost[arc_id_switch(net)] - base[arc_id_switch(net)] == pytest.approx(0.5)


def test_mu_zero_leaves_costs_unchanged():
    dets, gt = switch_scene()
    net = build_network(dets, gt, CostParams())
    before = net.arc_cost.copy()
    apply_params(net, CostParams(mu=0.0))
    assert np.array_equal(net.arc_cost, before)


def test_objective_breakdown_splits_id_switch():
    dets, gt = switch_scene()
    p = CostParams(gamma=0.5, mu=0.001)
    net = build_network(dets, gt, p)
    bodies, flow, info = chosen(net, p)
    assert bodies == [0, 0, 1, 0, 0]
    o = objective_breakdown(net, flow, p)
    assert o["n_id_switches"] == 2
    assert o["id_switch"] == pytest.approx(0.002)
    parts = ("pose", "reprojection", "false_positive", "false_negative",
             "temporal_and_gaps", "displacement", "id_switch", "boundary")
    assert sum(o[k] for k in parts) == pytest.approx(info["objective"])
    o0 = objective_breakdown(net, flow, CostParams(gamma=0.5))
    assert o0["id_switch"] == 0.0 and o0["n_id_switches"] == 2


def test_per_frame_baseline_ignores_mu_but_path_baseline_uses_it():
    """C_ID là chi phí cung: B2 (theo frame) không đổi, B3 (đường đi) đổi theo."""
    dets, gt = switch_scene()
    net = build_network(dets, gt, CostParams(gamma=0.0))
    body = {n["id"]: n["body_id"] for n in net.nodes}

    def bodies(assign):
        return [body[assign[(f, "P")]] for f in range(5)]

    apply_params(net, CostParams(gamma=0.0))
    b2, b3 = bodies(hungarian_per_frame(net)), bodies(independent_dp(net))
    apply_params(net, CostParams(gamma=0.0, mu=1.0))
    assert bodies(hungarian_per_frame(net)) == b2 == [0, 0, 1, 0, 0]
    assert b3 == [0, 0, 1, 0, 0]
    assert bodies(independent_dp(net)) == [0, 0, 0, 0, 0]


def test_cli_and_round_file_know_mu():
    assert OVERRIDES["mu"] == "mu"
    args = tune.parse_args(["--mu", "0.3", "--grid", "mu=0,1"])
    assert load_params(args.config, args).mu == 0.3
    assert tune.parse_grid(args.grid)["mu"] == [0.0, 1.0]
    p = CostParams(alpha=1.5, eta=0.1, beta=0.25, mu=2.0)
    assert CostParams.from_config(tune.association_block(p)) == p
    assert CostParams.from_config({}).mu == 0.0       # config cũ không có id_switch
