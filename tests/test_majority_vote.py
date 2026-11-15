"""B5 (A3 trong docs/MCMCF_revised.docx): B2 + majority vote nhãn theo body_id trong ±ω frame.
B5': như B5 nhưng vote dọc chuỗi nối bằng độ dịch 2D (≤ τ), không dùng body_id.

Chạy:  python3 -m pytest -q tests/test_majority_vote.py
"""
import numpy as np
import pytest

from pose_assoc.baselines import (VOTE_WINDOW, hungarian_majority_vote,
                                  hungarian_per_frame, hungarian_vote_2d_tracks,
                                  link_distance, link_tracks_2d, majority_vote_labels)
from pose_assoc.costs import CostParams
from pose_assoc.network import build_network


def labels_of(body_id, seq, start=0):
    return {(start + i, body_id): lab for i, lab in enumerate(seq)}


def test_isolated_flip_is_voted_away():
    labels = labels_of(0, ["A", "A", "A", "B", "A", "A", "A"])
    out = majority_vote_labels(labels, window=2)
    assert set(out.values()) == {"A"}


def test_window_zero_keeps_labels():
    labels = labels_of(0, ["A", "B", None, "B"])
    assert majority_vote_labels(labels, window=0) == labels


def test_none_is_a_label():
    out = majority_vote_labels(labels_of(1, [None, None, "A", None, None]), window=2)
    assert out[(2, 1)] is None


def test_tie_keeps_own_label():
    out = majority_vote_labels(labels_of(0, ["A", "B"]), window=1)
    assert out == {(0, 0): "A", (1, 0): "B"}


def test_window_counts_frames_not_occurrences():
    labels = {(0, 0): "A", (1, 0): "A", (50, 0): "B"}
    assert majority_vote_labels(labels, window=5)[(50, 0)] == "B"
    assert majority_vote_labels(labels, window=50)[(50, 0)] == "A"


def test_bodies_are_voted_separately():
    labels = {**labels_of(0, ["A", "A", "A"]), **labels_of(1, ["B", "A", "B"])}
    out = majority_vote_labels(labels, window=1)
    assert out[(1, 0)] == "A" and out[(1, 1)] == "B"


def test_negative_window_rejected():
    with pytest.raises(ValueError):
        majority_vote_labels({(0, 0): "A"}, window=-1)


def swap_scene():
    """2 người A, B; body 0 là A, body 1 là B ở mọi frame, trừ frame 3 WHAM đảo
    pose (body 0 giống B, body 1 giống A), nên B2 gán đảo ở frame 3."""
    rng = np.random.default_rng(1)
    pose_a, pose_b = rng.normal(scale=0.3, size=(12, 3)), rng.normal(scale=0.3, size=(12, 3))
    gt = {f: {"A": pose_a, "B": pose_b} for f in range(7)}
    noise = lambda: rng.normal(scale=0.005, size=(12, 3))
    dets = {f: [(0, pose_a + noise()), (1, pose_b + noise())] for f in range(7)}
    dets[3] = [(0, pose_b + noise()), (1, pose_a + noise())]
    return build_network(dets, gt, CostParams())


def chosen_bodies(net, assignment):
    return {k: (None if v is None else net.nodes[v]["body_id"]) for k, v in assignment.items()}


def test_b5_fixes_single_frame_swap_that_b2_makes():
    net = swap_scene()
    b2 = chosen_bodies(net, hungarian_per_frame(net))
    assert (b2[(3, "A")], b2[(3, "B")]) == (1, 0)          # B2 đảo ở frame 3
    b5 = chosen_bodies(net, hungarian_majority_vote(net, window=2))
    assert all(b5[(f, "A")] == 0 and b5[(f, "B")] == 1 for f in range(7))


def test_b5_window_zero_equals_b2():
    net = swap_scene()
    assert hungarian_majority_vote(net, window=0) == hungarian_per_frame(net)


def test_b5_never_shares_a_detection():
    net = swap_scene()
    b5 = chosen_bodies(net, hungarian_majority_vote(net, window=VOTE_WINDOW))
    for f in range(7):
        used = [b for (frame, _), b in b5.items() if frame == f and b is not None]
        assert len(used) == len(set(used))


# ---------------- B5': nối frame bằng độ dịch 2D ----------------

GRID = np.array([[x, y] for x in (0.0, 50.0, 100.0) for y in (0.0, 50.0, 100.0, 150.0)])


def uv_at(x, y=0.0):
    """12 khớp 2D của một người cao khoảng 150 px, góc trái trên ở (x, y)."""
    return GRID + np.array([x, y])


def test_link_distance_needs_enough_joints():
    a, b = uv_at(0), uv_at(3)
    assert link_distance(a, b) > 0
    few = b.copy()
    few[:5] = np.nan                                   # còn 7 khớp < 8
    assert np.isnan(link_distance(a, few))
    assert np.isnan(link_distance(a, None))


def test_link_follows_position_across_wham_id_switch():
    """Ở frame 3 WHAM đổi body_id của hai người; chuỗi vẫn bám theo vị trí."""
    det_uv = {}
    for f in range(6):
        left, right = (0, 1) if f < 3 else (1, 0)
        det_uv[(f, left)] = uv_at(2.0 * f)
        det_uv[(f, right)] = uv_at(400 + 2.0 * f)
    track = link_tracks_2d(list(det_uv), det_uv, threshold=0.2)
    left_tracks = {track[(f, 0 if f < 3 else 1)] for f in range(6)}
    right_tracks = {track[(f, 1 if f < 3 else 0)] for f in range(6)}
    assert len(left_tracks) == 1 and len(right_tracks) == 1
    assert left_tracks != right_tracks


def test_link_breaks_on_jump_and_divides_by_gap():
    det_uv = {(0, 0): uv_at(0), (1, 0): uv_at(200)}      # nhảy xa hơn τ -> chuỗi mới
    track = link_tracks_2d(list(det_uv), det_uv, threshold=0.2)
    assert track[(0, 0)] != track[(1, 0)]
    # dịch 0.3 cỡ người sau 2 frame: 0.3 / 2 = 0.15 <= 0.2 -> nối; gap_power = 0 -> không
    d = link_distance(uv_at(0), uv_at(1))
    shift = 0.3 / d
    det_uv = {(0, 0): uv_at(0), (2, 0): uv_at(shift)}
    assert link_tracks_2d(list(det_uv), det_uv, 0.2, gap_power=1.0)[(2, 0)] == 0
    assert link_tracks_2d(list(det_uv), det_uv, 0.2, gap_power=0.0)[(2, 0)] == 1
    assert link_tracks_2d(list(det_uv), det_uv, float("inf"), gap_power=0.0)[(2, 0)] == 0


def test_link_rejects_negative_threshold():
    with pytest.raises(ValueError):
        link_tracks_2d([(0, 0)], {(0, 0): uv_at(0)}, threshold=-1)


def scene_uv():
    """Toạ độ 2D cho swap_scene: body 0 luôn ở bên trái, body 1 ở bên phải."""
    return {(f, b): uv_at(400.0 * b + f) for f in range(7) for b in (0, 1)}


def test_b5_prime_fixes_single_frame_swap():
    net = swap_scene()
    b5p = chosen_bodies(net, hungarian_vote_2d_tracks(net, scene_uv(), window=2,
                                                      threshold=0.2))
    assert all(b5p[(f, "A")] == 0 and b5p[(f, "B")] == 1 for f in range(7))


def test_b5_prime_window_zero_equals_b2():
    net = swap_scene()
    assert hungarian_vote_2d_tracks(net, scene_uv(), window=0) == hungarian_per_frame(net)


def test_b5_prime_without_positions_is_b2():
    """Thiếu toạ độ thì không nối được gì: mỗi detection là một chuỗi, vote không đổi gì."""
    net = swap_scene()
    assert hungarian_vote_2d_tracks(net, {}, window=5) == hungarian_per_frame(net)
