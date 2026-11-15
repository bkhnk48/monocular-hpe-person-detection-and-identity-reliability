"""Chấm theo mức che trong pose_assoc.compare: MaxIoU khung 2D GT giữa hai người,
tầng tách / chồng / chồng nặng / không rõ, và số đếm theo tầng khớp với tổng.

Chạy:  python3 -m pytest -q tests/test_occlusion_strata.py
"""
import numpy as np
import pandas as pd
import pytest

from pose_assoc.baselines import hungarian_per_frame
from pose_assoc import build_network as bn
from pose_assoc.compare import _occlusion_strata, _print_strata
from pose_assoc.costs import CostParams
from pose_assoc.evaluation import (HEAVY_IOU, STRATA, box_iou, evaluate_assignment,
                                   joint_box, occlusion_strata, stratified_rows,
                                   stratum_of, unit_outcomes)
from pose_assoc.network import build_network

RNG = np.random.default_rng(0)
SQUARE = np.array([[x, y] for x in (0.0, 50.0, 100.0) for y in (0.0, 50.0, 100.0, 25.0)])


def body(x, y, scale=1.0):
    """12 khớp 2D có khung bao đúng [x, x + 100s] × [y, y + 100s]."""
    return SQUARE * scale + np.array([x, y])


def test_joint_box_and_iou():
    assert joint_box(body(10, 20)) == pytest.approx((10, 20, 110, 120))
    few = body(0, 0)
    few[:5] = np.nan                                   # còn 7 khớp < 8
    assert joint_box(few) is None
    a = joint_box(body(0, 0))
    assert box_iou(a, joint_box(body(200, 0))) == 0.0
    assert box_iou(a, joint_box(body(100, 0))) == 0.0  # chạm cạnh = không chồng
    assert box_iou(a, joint_box(body(50, 0))) == pytest.approx(5000 / 15000)
    assert box_iou(a, a) == pytest.approx(1.0)


def test_strata_thresholds_follow_ochuman():
    assert HEAVY_IOU == 0.5
    assert stratum_of(0.0) == "tách"
    assert stratum_of(0.2) == "chồng"
    assert stratum_of(0.5) == "chồng"
    assert stratum_of(0.51) == "chồng nặng"
    assert stratum_of(float("nan")) == "không rõ"


def scene():
    """2 người GT A, B; frame 0 tách, 1 chồng, 2 chồng nặng, 3 thiếu GT 2D của B,
    4 chỉ có A. body 0 ở gần A, body 1 ở gần B."""
    pose_a, pose_b = RNG.normal(scale=0.3, size=(12, 3)), RNG.normal(scale=0.3, size=(12, 3))
    gt = {f: {"A": pose_a, "B": pose_b} for f in range(4)}
    gt[4] = {"A": pose_a}
    dets = {f: [(0, pose_a + RNG.normal(scale=0.01, size=(12, 3))),
                (1, pose_b + RNG.normal(scale=0.01, size=(12, 3)))] for f in range(5)}
    shift_b = {0: 300, 1: 70, 2: 10}
    gt_uv = {}
    for f in range(5):
        gt_uv[(f, "A")] = body(0, 0)
        if f in shift_b:
            gt_uv[(f, "B")] = body(shift_b[f], 0)
    truth = {(f, 0): "A" for f in range(5)}
    truth.update({(f, 1): "B" for f in range(4)})
    truth[(4, 1)] = None                               # frame 4: body 1 không phải người GT
    return build_network(dets, gt, CostParams()), gt_uv, truth


def test_occlusion_strata_per_frame_and_person():
    net, gt_uv, _ = scene()
    st = occlusion_strata(net, gt_uv)
    assert {k: v[1] for k, v in st.items()} == {
        (0, "A"): "tách", (0, "B"): "tách",
        (1, "A"): "chồng", (1, "B"): "chồng",
        (2, "A"): "chồng nặng", (2, "B"): "chồng nặng",
        (3, "A"): "không rõ", (3, "B"): "không rõ",
        (4, "A"): "tách"}                              # chỉ một người trong frame
    assert st[(1, "A")][0] == pytest.approx(30 / 170)
    assert st[(2, "B")][0] == pytest.approx(90 / 110)


def test_stratified_counts_add_up_to_totals():
    net, gt_uv, truth = scene()
    assignment = hungarian_per_frame(net)
    total = evaluate_assignment(net, assignment, truth, method="B2", sequence="s")
    rows = stratified_rows(net, assignment, truth, occlusion_strata(net, gt_uv),
                           method="B2", sequence="s")
    assert [r["stratum"] for r in rows] == [s for s in STRATA if s in {r["stratum"] for r in rows}]
    for key in ("correct", "wrong_id", "missed", "correct_miss", "id_changes"):
        assert sum(r[key] for r in rows) == total[key], key
    assert sum(r["n_units"] for r in rows) == total["n_units"] == 9
    outcomes = unit_outcomes(net, assignment, truth)
    assert [(o["frame"], o["person"]) for o in outcomes] == sorted(
        (o["frame"], o["person"]) for o in outcomes)


def test_wrong_and_missed_land_in_the_right_stratum():
    net, gt_uv, truth = scene()
    nodes = {(n["frame"], n["person"], n["body_id"]): n["id"] for n in net.nodes
             if n["kind"] == "det"}
    # đúng hết, trừ frame 2 (chồng nặng) gán đảo người và frame 1 bỏ sót B
    assignment = {(f, p): nodes[(f, p, b)] for f in range(4) for p, b in (("A", 0), ("B", 1))}
    assignment[(4, "A")] = nodes[(4, "A", 0)]
    assignment[(2, "A")], assignment[(2, "B")] = nodes[(2, "A", 1)], nodes[(2, "B", 0)]
    assignment[(1, "B")] = None
    rows = {r["stratum"]: r for r in stratified_rows(net, assignment, truth,
                                                     occlusion_strata(net, gt_uv))}
    assert rows["chồng nặng"]["wrong_id"] == 2 and rows["chồng nặng"]["accuracy"] == 0.0
    assert rows["chồng"]["missed"] == 1 and rows["chồng"]["accuracy"] == pytest.approx(0.5)
    assert rows["tách"]["accuracy"] == 1.0 and rows["không rõ"]["accuracy"] == 1.0
    # đổi ID tính ở frame có body_id mới: frame 2 (đổi sang body kia) và frame 3 (đổi lại)
    assert rows["chồng nặng"]["id_changes"] == 2 and rows["không rõ"]["id_changes"] == 2


def test_print_strata_pools_sequences(capsys):
    rows = []
    for seq, (c2, c4) in {"s1": (8, 10), "s2": (1, 2)}.items():
        for method, c in (("B6", 5), ("B1", c2), ("B2", c2), ("B3", c2), ("B4", c4)):
            rows.append({"method": method, "sequence": seq, "stratum": "chồng", "n_units": 10,
                         "correct": c, "wrong_id": 10 - c, "missed": 0, "correct_miss": 0,
                         "id_changes": 1, "accuracy": c / 10})
    _print_strata(pd.DataFrame(rows))
    out = capsys.readouterr().out
    assert "THEO MỨC CHE" in out
    line = next(l for l in out.splitlines() if l.strip().startswith("chồng"))
    assert "0.4500" in line and "0.6000" in line and "0.1500" in line   # 9/20, 12/20, chênh


def test_broken_gt_2d_skips_strata_instead_of_stopping(monkeypatch, capsys):
    """beta = 0: GT 2D chỉ để chia tầng; zip hỏng thì bỏ phần chia tầng của sequence đó."""
    class Broken:
        def resolve_gt_2d(self, seq):
            raise ValueError("GT 2D poses2d.zip không có thư mục cam01")

    monkeypatch.setattr(bn, "mapping_module", lambda: Broken())
    net, _, _ = scene()
    assert _occlusion_strata({"name": "h4d_x_cam01"}, net) is None
    assert "không có thư mục cam01" in capsys.readouterr().out
