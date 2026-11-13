"""C_V (eta): độ dịch 2D giữa 2 detection liên tiếp, WHAM-2D.csv và chốt chặn cache của tune.

Chạy:  python3 -m pytest -q tests/test_displacement.py
"""
import pickle

import joblib
import numpy as np
import pytest

from pose_assoc import tune
from pose_assoc.build_network import mapping_module, positions_from_2d
from pose_assoc.costs import (LIMBS, CostParams, displacement_2d, displacement_cost,
                              person_scale_2d)
from pose_assoc.evaluation import assignment_table, objective_breakdown
from pose_assoc.network import apply_params, attach_positions, build_network
from pose_assoc.solvers.multi_mcf import solve_network

MOD = mapping_module()
RNG = np.random.default_rng(0)
TEMPLATE = RNG.uniform(0, 100, size=(12, 2))          # khung người 2D cỡ ~100 px
POSE = RNG.normal(scale=0.3, size=(12, 3))            # pose GT 3D (m)
OTHER = RNG.normal(scale=0.3, size=(12, 3))           # pose của một người khác


def at(x, y):
    return TEMPLATE + np.array([x, y], float)


# ------------------------------------------------------------------ công thức


def test_displacement_is_rms_shift_over_person_size():
    a = at(500, 500)
    assert displacement_2d(a, a) == 0.0
    d = displacement_2d(a, a + [10.0, 0.0])
    assert d == pytest.approx(10.0 / person_scale_2d(a))


def test_displacement_zero_without_enough_joints():
    a, b = at(0, 0), at(300, 0)
    assert displacement_2d(None, b) == 0.0
    b[:5] = np.nan                                     # còn 7 khớp < 8
    assert displacement_2d(a, b) == 0.0


def test_displacement_cost_divides_by_gap_like_motion_cost():
    p = CostParams(eta=2.0, disp_scale=0.05, gap_power=1.0)
    assert displacement_cost([0.1, 0.1], p, [1, 2]) == pytest.approx([4.0, 2.0])
    assert displacement_cost([0.1], CostParams(), [1]) == pytest.approx([0.0])   # eta = 0


# ------------------------------------------------------------------ mạng nhỏ


def switch_scene():
    """1 người GT, 5 frame. body 0 = đúng người (PA nhỏ, đi chậm 5 px/frame từ x = 300).
    body 1 = người khác ở x = 1500 px, riêng frame 2 có pose trùng hẳn GT (PA = 0),
    nên chỉ xét pose thì đường đi sẽ nhảy sang body 1 ở frame 2."""
    rng = np.random.default_rng(1)
    gt = {f: {"P": POSE} for f in range(5)}
    dets, uv = {}, {}
    for f in range(5):
        near = POSE + rng.normal(scale=0.01, size=(12, 3))
        dets[f] = [(0, near), (1, POSE.copy() if f == 2 else OTHER)]
        uv[(f, 0)], uv[(f, 1)] = at(300 + 5 * f, 400), at(1500, 400)
    return dets, gt, uv


def chosen(net, p):
    apply_params(net, p)
    flow, info = solve_network(net)
    rows = assignment_table(net, flow).sort_values("frame")
    return [int(b) for b in rows["body_id"].fillna(-1)], flow, info   # -1 = missed


def test_eta_keeps_identity_that_pose_alone_switches():
    dets, gt, uv = switch_scene()
    net = attach_positions(build_network(dets, gt, CostParams(gamma=0.0)), uv)
    assert chosen(net, CostParams(gamma=0.0))[0] == [0, 0, 1, 0, 0]
    # body 0 đi 5 px/frame (~0.05 cỡ người): eta = 0.2 phạt ~0.2/bước, nhảy sang body 1 ~48
    assert chosen(net, CostParams(gamma=0.0, eta=0.2))[0] == [0, 0, 0, 0, 0]


def test_arc_disp_only_on_dd_arcs():
    dets, gt, uv = switch_scene()
    net = attach_positions(build_network(dets, gt, CostParams()), uv)
    disp, sub = np.asarray(net.arc_disp), np.asarray(net.arc_sub)
    assert len(disp) == len(sub)
    assert np.all(disp[sub != "dd"] == 0.0)
    assert disp[sub == "dd"].max() > 5.0               # nhảy 1200 px với người ~100 px


def test_eta_zero_leaves_costs_unchanged():
    dets, gt, uv = switch_scene()
    p = CostParams()
    net = build_network(dets, gt, p)
    before = net.node_costs.copy(), net.arc_cost.copy()
    apply_params(attach_positions(net, uv), p)
    assert np.array_equal(net.node_costs, before[0])
    assert np.array_equal(net.arc_cost, before[1])


def test_eta_without_positions_is_skipped_not_crashing():
    """build_network() tự gọi apply_params() ở cuối, trước attach_positions()."""
    dets, gt, _ = switch_scene()
    plain = build_network(dets, gt, CostParams())
    net = build_network(dets, gt, CostParams(eta=1.0))
    assert np.array_equal(net.arc_cost, plain.arc_cost)
    net.arc_disp = [0.0]                               # lệch số cung -> phải báo lỗi
    with pytest.raises(ValueError, match="arc_disp"):
        apply_params(net, CostParams(eta=1.0))


def test_objective_breakdown_splits_displacement_without_double_count():
    dets, gt, uv = switch_scene()
    p = CostParams(gamma=0.5, eta=0.3)                 # body 0 đi 5 px/frame -> C_V > 0
    net = attach_positions(build_network(dets, gt, p), uv)
    _, flow, info = chosen(net, p)
    o = objective_breakdown(net, flow, p)
    assert o["displacement"] > 0
    parts = ("pose", "false_positive", "false_negative", "temporal_and_gaps",
             "displacement", "boundary")
    assert sum(o[k] for k in parts) == pytest.approx(info["objective"])


# ------------------------------------------------------------------ WHAM-2D.csv


def fake_pkl(tmp_path, monkeypatch):
    """pkl WHAM giả: J = đơn vị trên 24 'đỉnh' nên khớp SMPL = chính verts."""
    monkeypatch.setattr(MOD, "load_j_regressor", lambda path: np.eye(24))
    verts = RNG.uniform(-1, 1, size=(2, 24, 3))
    verts[..., 2] = RNG.uniform(2, 4, size=(2, 24))   # đứng trước camera
    path = tmp_path / "wham.pkl"
    joblib.dump({3: {"frame_ids": [5, 6], "verts": verts}}, path)
    return path, verts


def test_wham_pkl_to_2d_uses_wham_camera(tmp_path, monkeypatch):
    path, verts = fake_pkl(tmp_path, monkeypatch)
    W, H = 200, 100
    df = MOD.wham_pkl_to_2d(path, tmp_path / "smpl.pkl", (W, H))
    assert list(df.columns) == ["frame", "body_id"] + [f"{j}_{a}" for j in LIMBS
                                                       for a in "uv"]
    assert df["frame"].tolist() == [5, 6] and set(df["body_id"]) == {3}
    f = np.hypot(W, H)
    x, y, z = verts[1, MOD.WHAM_SMPL_IDX["left_shoulder"]]
    row = df.iloc[1]
    assert row["left_shoulder_u"] == pytest.approx(f * x / z + W / 2)
    assert row["left_shoulder_v"] == pytest.approx(f * y / z + H / 2)
    uv = positions_from_2d(df)
    assert set(uv) == {(5, 3), (6, 3)} and uv[(6, 3)].shape == (12, 2)
    assert uv[(6, 3)][0] == pytest.approx([row["left_shoulder_u"], row["left_shoulder_v"]])


def test_load_wham_2d_needs_image_size_then_caches(tmp_path, monkeypatch):
    path, _ = fake_pkl(tmp_path, monkeypatch)
    seq = {"name": "s", "wham_pkl": path, "wham_pkl_is_drive": False,
           "out_dir": tmp_path / "out"}
    with pytest.raises(ValueError, match="image_size"):
        MOD.load_wham_2d(seq, tmp_path / "smpl.pkl")
    seq["image_size"] = [200, 100]
    first = MOD.load_wham_2d(seq, tmp_path / "smpl.pkl")
    assert (seq["out_dir"] / "WHAM-2D.csv").is_file()
    path.unlink()                                      # lần sau đọc CSV, không cần pkl
    again = MOD.load_wham_2d(seq, tmp_path / "smpl.pkl")
    assert np.allclose(first.to_numpy(float), again.to_numpy(float))


# ------------------------------------------------------------------ tune


def labeled_seq(tmp_path):
    out = tmp_path / "seq"
    out.mkdir()
    (out / "mapping_frames.csv").write_text("frame,body_id,person\n0,1,S1\n")
    gt = tmp_path / "gt.csv"
    gt.write_text("frame,person\n0,S1\n")
    return {"name": "seq", "wham_pkl": "x" * 25, "wham_pkl_is_drive": True,
            "out_dir": out, "gt": {"type": "mpi_overlay_csv", "path": gt}}


def test_classify_rebuilds_entry_from_old_network(tmp_path):
    seq = labeled_seq(tmp_path)
    key = tune.seq_key(seq)
    entry = {"name": "seq", "mapping_sig": tune.mapping_signature(seq),
             "gt_signature": tune.gt_signature(seq), "excluded": False, "added_round": 1}
    assert tune.classify(seq, key, {"sequences": {key: dict(entry)}}, set()) == "outdated"
    current = dict(entry, net_version=tune.NET_VERSION)
    assert tune.classify(seq, key, {"sequences": {key: current}}, set()) == "known"
    assert tune.classify(dict(seq, skip=True), key, {"sequences": {key: entry}}, set()) == "skip"


def test_load_cached_refuses_to_mix_old_networks(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "CACHE_DIR", tmp_path)
    caches = {"drive:new": {"name": "moi", "net_version": tune.NET_VERSION},
              "drive:old": {"name": "cu"}}
    for key, d in caches.items():
        with open(tune.cache_path(key), "wb") as f:
            pickle.dump(d, f)
    reg = {"sequences": {k: {"name": d["name"]} for k, d in caches.items()}}
    with pytest.raises(SystemExit, match="--exclude cu"):
        tune.load_cached(reg)
    reg["sequences"]["drive:old"]["excluded"] = True
    assert [d["name"] for d in tune.load_cached(reg)] == ["moi"]


def test_round_file_keeps_eta():
    p = CostParams(alpha=1.5, eta=0.75, disp_scale=0.1)
    assert CostParams.from_config(tune.association_block(p)) == p
