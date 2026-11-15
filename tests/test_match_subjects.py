"""GT dạng CSV (wham_gt_mapping.csv_gt_poses, gt.type = "csv") và gợi ý + gán tay mapping
(scripts/match_wham_subjects_to_gt.py) trên dữ liệu tổng hợp nhỏ, không cần pkl/video."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import match_wham_subjects_to_gt as mw  # noqa: E402

MOD = mw.MOD
JOINTS = list(MOD.MPI_IDX)          # neck + 12 khớp chi


def pose_a() -> np.ndarray:
    """13 khớp (m): người đứng, tay dang ngang."""
    p = np.zeros((13, 3))
    p[0] = (0, -0.5, 0)                                  # neck
    for i, j in enumerate(JOINTS[1:], 1):
        side = -1 if j.startswith("left") else 1
        part = j.split("_")[1]
        p[i] = {"shoulder": (0.2 * side, -0.45, 0), "elbow": (0.45 * side, -0.45, 0.02),
                "wrist": (0.7 * side, -0.45, 0.05), "hip": (0.1 * side, 0.0, 0),
                "knee": (0.12 * side, 0.45, 0.03), "ankle": (0.12 * side, 0.9, 0)}[part]
    return p


def pose_b() -> np.ndarray:
    """Tay giơ cao, một gối co: hình dạng khác hẳn pose_a."""
    p = pose_a().copy()
    for i, j in enumerate(JOINTS):
        if "elbow" in j:
            p[i, 1] = -0.8
        if "wrist" in j:
            p[i, 1], p[i, 0] = -1.1, p[i, 0] * 0.4
        if j == "left_knee":
            p[i] = (-0.1, 0.3, 0.35)
        if j == "left_ankle":
            p[i] = (-0.1, 0.6, 0.1)
    return p


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gt_rows(frame, person, p13_m, shift_mm=(0, 0, 0)):
    row = {"frame": frame, "person": person, "src_frame": frame}
    for name, v in zip(JOINTS, p13_m * 1000.0 + np.asarray(shift_mm)):
        row.update({f"{name}_x": v[0], f"{name}_y": v[1], f"{name}_z": v[2]})
    return row


def wham_row(frame, bid, p13_m, angle=0.3, scale=0.9, noise=0.0):
    """Detection WHAM: cùng hình dạng nhưng xoay, co giãn, dời gốc, trục khác (như RA-WHAM);
    noise (m) làm lệch hình dạng."""
    p13_m = p13_m + np.random.default_rng(frame * 10 + bid).normal(scale=noise, size=(13, 3))
    q = scale * (p13_m - p13_m[0]) @ rot_y(angle).T + (1.0, 0.0, 3.0)
    row = {"frame_id": frame, "body_id": bid, "pelvis_x": 0.0, "pelvis_y": 0.0, "pelvis_z": 0.0}
    for name, v in zip(JOINTS[1:], q[1:]):     # (-x, z, y) như wham_subject_table: det = +1
        row.update({f"{name}_x": -v[0], f"{name}_y": v[2], f"{name}_z": v[1]})
    return row


@pytest.fixture
def clip(tmp_path):
    """10 frame, 2 người (X = pose_a, Y = pose_b). WHAM: body 0 là X suốt; body 1 là Y ở
    frame 0-5 rồi nhảy sang X ở 6-9 (track nhảy người, kém khớp hơn body 0); body 2 là
    pose lạ."""
    a, b = pose_a(), pose_b()
    weird = np.random.default_rng(0).normal(size=(13, 3))
    gt, wham = [], []
    for f in range(10):
        gt += [gt_rows(f, "X_S8", a), gt_rows(f, "Y_S8", b, shift_mm=(-600, 0, 3400))]
        wham.append(wham_row(f, 0, a))
        wham.append(wham_row(f, 1, b if f <= 5 else a, angle=-0.5, noise=0.0 if f <= 5 else 0.03))
        if f >= 3:
            wham.append(wham_row(f, 2, weird))
    gdir = tmp_path / "ovl_test"
    gdir.mkdir()
    pd.DataFrame(gt).to_csv(gdir / "gt_3d.csv", index=False)
    pd.DataFrame(wham).to_csv(tmp_path / "RA-WHAM.csv", index=False)
    return gdir, tmp_path / "RA-WHAM.csv", tmp_path / "out"


# ------------------------------------------------------------ đọc GT CSV


def test_csv_gt_multi_person_units_and_order(clip):
    gdir, _, _ = clip
    poses = MOD.csv_gt_poses(gdir / "gt_3d.csv")
    assert sorted(poses) == list(range(10))
    assert set(poses[0]) == {"X_S8", "Y_S8"}
    assert poses[0]["X_S8"].shape == (13, 3)
    np.testing.assert_allclose(poses[0]["X_S8"], pose_a(), atol=1e-9)   # mm -> m
    gt, skel = MOD.load_gt({"type": "csv", "path": gdir / "gt_3d.csv"})
    assert skel == "mpi13" and gt.keys() == poses.keys()


def test_csv_gt_single_person_legacy_with_bom(tmp_path):
    row = gt_rows(7, "ignored", pose_a())
    del row["person"], row["src_frame"]
    path = tmp_path / "S8_Seq1_video_cam0.csv"
    pd.DataFrame([{"video": "video_0", "camera": 0, **row}]).to_csv(
        path, index=False, encoding="utf-8-sig")
    poses = MOD.csv_gt_poses(path, person="S8")
    assert list(poses) == [7] and list(poses[7]) == ["S8"]


def test_csv_gt_rejects_missing_columns_and_duplicates(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"frame": 0, "person": "X", "neck_x": 1}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="thiếu cột"):
        MOD.csv_gt_poses(bad)
    dup = tmp_path / "dup.csv"
    pd.DataFrame([gt_rows(0, "X", pose_a()), gt_rows(0, "X", pose_a())]).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="hai dòng"):
        MOD.csv_gt_poses(dup)


def test_scene_shift_does_not_change_pa(clip):
    """gt_3d_scene.csv = gt_3d.csv tịnh tiến nguyên khối -> PA-MPJPE y hệt."""
    gdir, wham_csv, _ = clip
    df = pd.read_csv(gdir / "gt_3d.csv")
    scene = df.copy()
    for c in scene.columns:
        if c.endswith("_z") and (scene["person"] == "Y_S8").any():
            scene.loc[scene["person"] == "Y_S8", c] += 3419.0
    scene.to_csv(gdir / "gt_3d_scene.csv", index=False)
    wham = pd.read_csv(wham_csv)
    a = mw.pa_scores(wham, MOD.csv_gt_poses(gdir / "gt_3d.csv"))
    b = mw.pa_scores(wham, MOD.csv_gt_poses(gdir / "gt_3d_scene.csv"))
    np.testing.assert_allclose(a["pa_mpjpe_mm"], b["pa_mpjpe_mm"], atol=1e-6)


# ------------------------------------------------------------ gợi ý


def test_pa_suggestion_follows_shape_and_track_switch(clip):
    gdir, wham_csv, _ = clip
    wham = pd.read_csv(wham_csv)
    tracks = MOD.track_frames(wham)
    scores = mw.pa_scores(wham, MOD.csv_gt_poses(gdir / "gt_3d.csv"))
    ok = scores[scores["body_id"] == 0].set_index(["frame", "person"])["pa_mpjpe_mm"]
    assert ok.xs("X_S8", level="person").max() < 1e-6        # cùng hình dạng -> PA = 0
    lab = mw.suggest_pa(scores, tracks, max_pa=150.0)
    assert set(lab[0].values()) == {"X_S8"}
    assert all(lab[1][f] == "Y_S8" for f in range(6))
    # frame 6-9: body 0 và 1 đều là X; Hungarian chỉ cho X một detection, body còn lại
    # khớp Y quá tệ (> max_pa) -> không phải người GT
    assert all(lab[1][f] == MOD.NOT_PERSON for f in range(6, 10))
    assert set(lab[2].values()) == {MOD.NOT_PERSON}


def test_read_mapping_roundtrip(tmp_path):
    tracks = {0: np.arange(0, 5), 1: np.arange(2, 9), 7: np.arange(0, 3)}
    mapping = {0: {f: "X" for f in range(5)},
               1: {**{f: "Y" for f in range(2, 5)}, **{f: "X" for f in range(5, 7)},
                   **{f: "" for f in range(7, 9)}},
               7: {f: "" for f in range(3)}}
    MOD.write_mapping(tmp_path, mapping, {"name": "t", "wham_pkl": "p", "gt": {"path": "g"}},
                      notes=["gợi ý: test"])
    text = (tmp_path / "mapping.txt").read_text(encoding="utf-8")
    assert "# gợi ý: test" in text
    assert MOD.read_mapping(tmp_path / "mapping.txt", tracks, ["X", "Y"]) == mapping


def test_read_mapping_errors(tmp_path):
    tracks = {0: np.arange(0, 5)}
    for body, msg in (("9 = X\n", "không có trong WHAM"), ("0 = Z\n", "không có người GT"),
                      ("0 = X @ 3-20\n", "ngoài track"), ("abc\n", "không hợp lệ")):
        (tmp_path / "mapping.txt").write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            MOD.read_mapping(tmp_path / "mapping.txt", tracks, ["X", "Y"])


# ------------------------------------------------------------ chương trình chính


def run(gdir, wham_csv, out, *extra):
    return mw.main(["--gt", str(gdir), "--wham", str(wham_csv), "--out", str(out), *extra])


def test_main_assign_overrides_and_header(clip, capsys):
    gdir, wham_csv, out = clip
    assert run(gdir, wham_csv, out, "--yes", "--assign", "1=Y_S8, X_S8@6-9") == 0
    assert "gợi ý theo PA-MPJPE" in capsys.readouterr().out         # cảnh báo vòng lặp
    frames = pd.read_csv(out / "mapping_frames.csv", dtype={"person": str}).fillna("")
    b1 = frames[frames["body_id"] == 1].set_index("frame")["person"]
    assert (b1.loc[0:5] == "Y_S8").all() and (b1.loc[6:9] == "X_S8").all()
    text = (out / "mapping.txt").read_text(encoding="utf-8")
    assert "gợi ý: pa_mpjpe" in text and "gán tay: body_id [1]" in text
    assert "CẢNH BÁO" in text
    assert "1 = Y_S8 @ 0-5" in text and "1 = X_S8 @ 6-9" in text
    for name in ("frame_bodyid_scores.csv", "subject_id_metrics.csv"):
        assert (out / name).is_file()


def test_main_restarts_from_hand_edited_mapping(clip):
    gdir, wham_csv, out = clip
    run(gdir, wham_csv, out, "--yes")
    path = out / "mapping.txt"
    path.write_text(path.read_text(encoding="utf-8") + "2 = Y_S8 @ 3-4\n", encoding="utf-8")
    run(gdir, wham_csv, out, "--yes")
    frames = pd.read_csv(out / "mapping_frames.csv", dtype={"person": str}).fillna("")
    b2 = frames[frames["body_id"] == 2].set_index("frame")["person"]
    assert list(b2.loc[3:4]) == ["Y_S8", "Y_S8"] and (b2.loc[5:] == "").all()
    run(gdir, wham_csv, out, "--yes", "--fresh")                    # bỏ bản sửa tay
    frames = pd.read_csv(out / "mapping_frames.csv", dtype={"person": str}).fillna("")
    assert (frames[frames["body_id"] == 2]["person"] == "").all()


def test_main_interactive_answers(clip, monkeypatch):
    gdir, wham_csv, out = clip
    answers = iter(["", "x", "2"])          # body 0: nhận gợi ý, body 1: không phải người, body 2: Y
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert run(gdir, wham_csv, out) == 0
    frames = pd.read_csv(out / "mapping_frames.csv", dtype={"person": str}).fillna("")
    by = frames.groupby("body_id")["person"].agg(set)
    assert by[0] == {"X_S8"} and by[1] == {""} and by[2] == {"Y_S8"}
    assert "gán tay: body_id [1, 2]" in (out / "mapping.txt").read_text(encoding="utf-8")


def test_main_quit_writes_nothing(clip, monkeypatch):
    gdir, wham_csv, out = clip
    monkeypatch.setattr("builtins.input", lambda *_: "q")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert run(gdir, wham_csv, out) == 1
    assert not (out / "mapping.txt").exists()


def test_main_without_terminal_needs_yes(clip, monkeypatch):
    gdir, wham_csv, out = clip
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="--yes"):
        run(gdir, wham_csv, out)
    with pytest.raises(SystemExit, match="body_id 9"):
        run(gdir, wham_csv, out, "--yes", "--assign", "9=X_S8")
    with pytest.raises(SystemExit, match="gt_2d.csv chỉ có 2D"):
        mw.main(["--gt", str(gdir / "gt_2d.csv"), "--wham", str(wham_csv)])
