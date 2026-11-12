"""Kiểm thử video ghép 2 người (scripts/make_overlay_pair.py) và gán nhãn tự động
(scripts/overlay_autolabel.py) trên dữ liệu tổng hợp nhỏ, không cần video/WHAM."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_overlay_pair as mk  # noqa: E402
import overlay_autolabel as al  # noqa: E402


def standing_skeleton(cx: float, top: float = 100.0, h: float = 400.0) -> np.ndarray:
    """28 khớp MPI 2D của một người đứng thẳng, tâm ngang cx."""
    j = np.zeros((28, 2))
    y = lambda r: top + r * h  # noqa: E731
    spine = {7: 0.0, 6: 0.06, 5: 0.15, 1: 0.22, 0: 0.3, 2: 0.38, 3: 0.45, 4: 0.5}
    for i, r in spine.items():
        j[i] = (cx, y(r))
    for side, sx in ((0, -1), (1, 1)):          # 0 = trái (8..12, 18..22), 1 = phải
        arm = [8, 9, 10, 11, 12] if side == 0 else [13, 14, 15, 16, 17]
        leg = [18, 19, 20, 21, 22] if side == 0 else [23, 24, 25, 26, 27]
        for k, (dxr, r) in enumerate([(0.05, 0.16), (0.1, 0.17), (0.12, 0.32),
                                      (0.13, 0.45), (0.13, 0.5)]):
            j[arm[k]] = (cx + sx * dxr * h, y(r))
        for k, (dxr, r) in enumerate([(0.05, 0.5), (0.05, 0.73), (0.05, 0.95),
                                      (0.05, 0.98), (0.05, 1.0)]):
            j[leg[k]] = (cx + sx * dxr * h, y(r))
    return j


def limbs(j28: np.ndarray) -> np.ndarray:
    return j28[al.MPI_LIMB_IDX]


def det_table(rows) -> pd.DataFrame:
    """rows: (frame, body_id, 12x2 điểm) -> bảng giống wham_2d()."""
    out = []
    for f, b, p in rows:
        r = {"frame": f, "body_id": b}
        for i, name in enumerate(al.LIMBS):
            r[f"{name}_u"], r[f"{name}_v"] = p[i]
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- gán nhãn tự động

def test_label_follows_2d_position_not_track_id():
    """Track 7 đi theo A rồi nhảy sang B (WHAM đổi ID): nhãn phải đổi theo vị trí."""
    A, B = limbs(standing_skeleton(500)), limbs(standing_skeleton(900))
    gt = {f: {"A_S8": A, "B_S8": B} for f in range(4)}
    dets = det_table([(0, 7, A + 3), (1, 7, A - 2), (2, 7, B + 1), (3, 7, B),
                      (0, 9, B + 2), (1, 9, B - 1), (2, 9, A), (3, 9, A + 4)])
    lab = al.label_detections(dets, gt).set_index(["frame", "body_id"])["person"]
    assert [lab[(f, 7)] for f in range(4)] == ["A_S8", "A_S8", "B_S8", "B_S8"]
    assert [lab[(f, 9)] for f in range(4)] == ["B_S8", "B_S8", "A_S8", "A_S8"]


def test_far_detection_is_not_a_gt_person():
    A = limbs(standing_skeleton(500))
    gt = {0: {"A_S8": A}}
    dets = det_table([(0, 1, A + (1500.0, 0.0))])
    assert al.label_detections(dets, gt)["person"].tolist() == [""]


def test_unsure_detection_recovered_only_inside_a_confident_track():
    A = limbs(standing_skeleton(500))
    off = (0.35 * 400, 0.0)                     # e ≈ 0.35: vùng không chắc
    gt = {f: {"A_S8": A} for f in range(3)}
    dets = det_table([(0, 1, A), (1, 1, A + off),            # cùng track có frame chắc chắn
                      (2, 2, A + off)])                     # track lẻ: không nhận
    lab = al.label_detections(dets, gt).set_index(["frame", "body_id"])
    assert lab.loc[(1, 1), "person"] == "A_S8" and lab.loc[(1, 1), "review"]
    assert lab.loc[(2, 2), "person"] == ""


def test_mapping_round_trip():
    lab = pd.DataFrame({"frame": [0, 1, 0], "body_id": [3, 3, 4],
                        "person": ["A_S8", "", "B_S8"]})
    assert al.to_mapping(lab) == {3: {0: "A_S8", 1: ""}, 4: {0: "B_S8"}}


# ---------------------------------------------------------------- tạo video ghép

def test_mask_keeps_person_and_drops_green_background():
    img = np.zeros((600, 1000, 3), np.uint8)
    img[:] = (40, 140, 40)                       # phông xanh (BGR)
    j = standing_skeleton(500)
    img[90:510, 440:560] = (200, 200, 200)       # "người" xám
    alpha, fg, (x0, y0, x1, y1) = mk.person_alpha(img, j, tau=12, width=0.16)
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = fg
    person = np.zeros(img.shape[:2], bool)
    person[90:510, 440:560] = True
    inside = full[person].mean()
    outside = full[~person].mean()
    assert inside > 0.6 and outside < 0.01


def test_depth_scale_ignores_pose_and_flags_bad_depth():
    K = {"fx": 1500.0, "fy": 1500.0, "cx": 1024.0, "cy": 1024.0}
    a3 = np.zeros((3, 28, 3))
    a3[:, :, 2] = 3000.0
    a3[0, :, 1] = np.linspace(-900, 900, 28)       # đứng: cao 1.8 m
    a3[1, :, 1] = np.linspace(-400, 400, 28)       # ngồi xổm: thấp hơn nửa, cùng độ sâu
    a3[2, mk.PELVIS, 2] = np.nan
    s = mk.depth_scale(a3, K, 1750.0)
    assert np.isclose(s[0], 1500 * 1750 / 3000) and s[0] == s[1]
    assert np.isnan(s[2])


def test_depth_scale_keeps_limbs_when_2d_box_shrinks():
    """Người ngồi xổm: khung 2D chỉ còn 200 px nhưng thân vẫn rộng như lúc đứng (cỡ 400)."""
    img = np.zeros((600, 1000, 3), np.uint8)
    img[:] = (40, 140, 40)
    j = standing_skeleton(500, top=100, h=200)
    img[100:300, 460:541] = (200, 200, 200)        # thân + đùi rộng 80 px
    person = np.zeros(img.shape[:2], bool)
    person[100:300, 460:541] = True

    def kept(scale):
        _, fg, (x0, y0, x1, y1) = mk.person_alpha(img, j, 12, 0.16, scale)
        full = np.zeros(img.shape[:2], bool)
        full[y0:y1, x0:x1] = fg
        return full[person].mean(), full[~person].mean()

    in_2d, _ = kept(None)
    in_nan, _ = kept(float("nan"))
    in_depth, out_depth = kept(400.0)
    assert in_nan == in_2d                         # NaN -> quay về khung 2D
    assert in_2d < 0.85 and in_depth > 0.95 and out_depth < 0.01


def test_paste_scaled_shrinks_about_center():
    """Hộp (60..80) thu nhỏ 0.5 quanh tâm (50, 50) -> (55..65)."""
    dst = np.zeros((100, 100, 3), np.uint8)
    src = np.full((100, 100, 3), 255, np.uint8)
    mask = mk.paste_scaled(dst, src, np.ones((20, 20), np.float32), (60, 60, 80, 80),
                           s=0.5, center=(50, 50))
    assert dst[55:65, 55:65].min() == 255 and mask.sum() == 100
    assert dst.sum() == 255 * 3 * 100
    np.testing.assert_allclose(mk.shrink((60, 80), 0.5, (50, 50)), (55, 65))


def test_paste_clips_at_border():
    dst = np.zeros((10, 10, 3), np.uint8)
    src = np.full((10, 10, 3), 255, np.uint8)
    mask = mk.paste_scaled(dst, src, np.ones((4, 4), np.float32), (6, 2, 10, 6),
                           s=2.0, center=(5, 5))    # -> x 7..15, y -1..7: cắt còn x 7..10, y 0..7
    assert dst[0:7, 7:].min() == 255 and mask[0:7, 7:].all()
    assert dst[:, :7].max() == 0 and dst[7:].max() == 0


def test_front_person_always_covers_back_person():
    """Thứ tự dán như composite(): Y (thu nhỏ) trước, X sau cùng -> chỗ chồng là X."""
    h = w = 400
    green = np.zeros((h, w, 3), np.uint8)
    green[:] = (40, 140, 40)
    img_x, img_y = green.copy(), green.copy()
    jx, jy = standing_skeleton(230, 60, 280), standing_skeleton(200, 0, 400)
    img_x[50:350, 200:260] = (200, 200, 200)     # X lệch phải: che một phần Y (x 175..225)
    img_y[0:400, 150:250] = (90, 60, 160)
    px = mk.person_alpha(img_x, jx, 12, 0.16)
    py = mk.person_alpha(img_y, jy, 12, 0.16)
    out = img_x.copy()
    my = mk.paste_scaled(out, img_y, py[0], py[2], 0.5, (w / 2, h / 2))
    mx = mk.paste_scaled(out, img_x, px[0], px[2])
    import cv2
    inner = cv2.erode(mx.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # bỏ viền làm mềm
    both = inner & my
    assert both.sum() > 1000
    assert (np.abs(out[both].astype(int) - 200) <= 2).all()            # chỗ chồng là màu của X
    y_only = (cv2.erode(my.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & \
        ~(cv2.dilate(mx.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    assert y_only.sum() > 1000                                         # phần Y không bị che vẫn thấy
    assert (np.abs(out[y_only].astype(int) - (90, 60, 160)) <= 2).all()


# ---------------------------------------------------------------- GT của video ghép

def test_overlay_gt_uses_source_frames_and_shrinks_y_in_2d(tmp_path):
    import json

    from scipy.io import savemat

    F = 6
    a3 = np.arange(F * 28 * 3, dtype=float).reshape(F, 84)          # mm, mỗi frame khác nhau
    a2 = np.arange(F * 28 * 2, dtype=float).reshape(F, 56)
    cams = lambda x: np.array([x, x * 0 - 1], dtype=object)          # 2 camera, tránh squeeze  # noqa: E731
    savemat(tmp_path / "annot.mat", {"annot3": cams(a3), "annot2": cams(a2)})
    pd.DataFrame({"frame": [0, 1], "src_x": [1, 2], "src_y": [4, 5]}).to_csv(
        tmp_path / "overlay_frames.csv", index=False)
    c = [1024.0, 1024.0]
    meta = {"persons": {
        "X_S8": {"layer": "front", "camera": 0, "annot": "annot.mat", "src_col": "src_x",
                 "scale": 1.0, "center": c},
        "Y_S8": {"layer": "back", "camera": 0, "annot": "annot.mat", "src_col": "src_y",
                 "scale": 0.5, "center": c}},
        "frames_csv": "overlay_frames.csv", "out_scale": 1.0, "size": [2048, 2048]}
    (tmp_path / "overlay_meta.json").write_text(json.dumps(meta))

    gt3 = al.MOD.overlay_gt_poses(tmp_path / "overlay_meta.json")
    idx = list(al.MOD.MPI_IDX.values())
    np.testing.assert_allclose(gt3[0]["X_S8"], a3[1].reshape(28, 3)[idx] / 1000)
    np.testing.assert_allclose(gt3[1]["Y_S8"], a3[5].reshape(28, 3)[idx] / 1000)

    gt2 = al.overlay_gt_2d(tmp_path / "overlay_meta.json")
    raw = a2.reshape(F, 28, 2)[:, al.MPI_LIMB_IDX]
    np.testing.assert_allclose(gt2[0]["X_S8"], raw[1])
    np.testing.assert_allclose(gt2[0]["Y_S8"], 1024 + 0.5 * (raw[4] - 1024))
    np.testing.assert_allclose(gt2[1]["Y_S8"], 1024 + 0.5 * (raw[5] - 1024))


def test_export_gt_files_are_consistent(tmp_path):
    """gt_3d = annot3 nguồn; gt_2d của Y = thu nhỏ; gt_3d_scene của Y chiếu đúng pelvis đã dán."""
    import json

    from scipy.io import savemat

    rng = np.random.default_rng(0)
    F, K = 6, {"fx": 1500.0, "fy": 1490.0, "cx": 1024.0, "cy": 1050.0}
    body = rng.normal(0, 250, (28, 3))
    a3 = body[None] + np.c_[rng.uniform(-800, 800, F), rng.uniform(-200, 600, F),
                            rng.uniform(3000, 4500, F)][:, None]
    a2 = mk.project(a3, K)
    cams = lambda x: np.array([x.reshape(F, -1), x.reshape(F, -1) * 0], dtype=object)  # noqa: E731
    savemat(tmp_path / "annot.mat", {"annot3": cams(a3), "annot2": cams(a2)})
    pd.DataFrame({"frame": [0, 1], "src_x": [1, 2], "src_y": [4, 5]}).to_csv(
        tmp_path / "overlay_frames.csv", index=False)
    c = [1024.0, 1024.0]
    meta = {"persons": {
        "X_S8": {"layer": "front", "camera": 0, "annot": "annot.mat", "src_col": "src_x",
                 "scale": 1.0, "center": c},
        "Y_S8": {"layer": "back", "camera": 0, "annot": "annot.mat", "src_col": "src_y",
                 "scale": 0.5, "center": c}},
        "frames_csv": "overlay_frames.csv", "out_scale": 1.0}
    (tmp_path / "overlay_meta.json").write_text(json.dumps(meta))

    gt = mk.export_gt(tmp_path)
    assert abs(gt["intrinsics_fit"]["X_S8"]["fx"] - 1500) < 1e-6
    assert abs(gt["intrinsics_fit"]["X_S8"]["cy"] - 1050) < 1e-6

    cols = lambda df, ax: df[[f"{j}_{a}" for j in mk.GT_JOINTS for a in ax]].to_numpy().reshape(  # noqa: E731
        len(df), len(mk.GT_JOINTS), len(ax))
    idx = list(mk.GT_JOINTS.values())
    g3 = pd.read_csv(tmp_path / "gt_3d.csv").set_index(["frame", "person"])
    g2 = pd.read_csv(tmp_path / "gt_2d.csv").set_index(["frame", "person"])
    gs = pd.read_csv(tmp_path / "gt_3d_scene.csv").set_index(["frame", "person"])
    y3, y2, ys = (g.xs("Y_S8", level="person") for g in (g3, g2, gs))
    np.testing.assert_allclose(cols(g3.xs("X_S8", level="person"), "xyz"), a3[[1, 2]][:, idx], atol=1e-3)
    np.testing.assert_allclose(cols(gs.xs("X_S8", level="person"), "xyz"), a3[[1, 2]][:, idx], atol=1e-3)
    np.testing.assert_allclose(cols(y3, "xyz"), a3[[4, 5]][:, idx], atol=1e-3)
    np.testing.assert_allclose(cols(y2, "uv"), 1024 + 0.5 * (a2[[4, 5]][:, idx] - 1024), atol=1e-3)
    scene = cols(ys, "xyz")
    np.testing.assert_allclose(mk.project(scene[:, 0], K), cols(y2, "uv")[:, 0], atol=1e-2)  # pelvis
    np.testing.assert_allclose(scene[:, 0, 2], 2 * a3[[4, 5], mk.PELVIS, 2], rtol=1e-6)     # xa gấp đôi
    np.testing.assert_allclose(scene - scene[:, :1], cols(y3, "xyz") - cols(y3, "xyz")[:, :1],
                               atol=1e-3)                                                 # cùng thân, chỉ tịnh tiến


# ---------------------------------------------------------------- chọn đoạn nguồn

def test_intervals_overlap_matches_review_table():
    n = 1500
    assert mk.intervals_overlap(2000, 2500, n, 100)          # giao nhau
    assert mk.intervals_overlap(2000, 3250, n, 100)          # chung 250 frame (lỗi cũ)
    assert mk.intervals_overlap(2000, 3500, n, 100)          # sát nhau, thiếu khoảng cách
    assert not mk.intervals_overlap(2000, 3600, n, 100)
    assert not mk.intervals_overlap(5000, 2000, n, 100)      # đối xứng
    assert not mk.intervals_overlap(2000, 3500, n, 0)


def test_same_timeline_ignores_camera(tmp_path):
    a = tmp_path / "annot.mat"
    a.write_bytes(b"")
    assert mk.same_timeline({"annot": a}, {"annot": tmp_path / "." / "annot.mat"})
    assert not mk.same_timeline({"annot": a}, {"annot": tmp_path / "other.mat"})


def test_touches_border():
    j = np.stack([standing_skeleton(500), standing_skeleton(5)])   # người thứ 2 tràn mép trái
    assert mk.touches_border(j, (2048, 2048)).tolist() == [False, True]


# ---------------------------------------------------------------- kiểm chứng theo mức che

def test_occlusion_bins_flag_heavy_frames_and_audit_scores(tmp_path):
    import json

    pd.DataFrame({"frame": [0, 1, 2], "vis_y": [1.0, 0.7, 0.2]}).to_csv(
        tmp_path / "overlay_frames.csv", index=False)
    (tmp_path / "overlay_meta.json").write_text(json.dumps({"frames_csv": "overlay_frames.csv"}))
    lab = pd.DataFrame({"frame": [0, 1, 2, 2], "body_id": [1, 1, 1, 2],
                        "person": ["X_S8", "X_S8", "Y_S8", ""], "e_best": [.03, .04, .1, .9],
                        "review": [False] * 4})
    lab = al.add_occlusion(lab, tmp_path / "overlay_meta.json")
    assert lab["occlusion"].tolist() == ["khong_che", "che_vua", "che_nang", "che_nang"]
    assert lab["review"].tolist() == [False, False, True, True]

    audit = al.audit_sample(lab, per_bin=5)
    audit["manual_person"] = ["X_S8", "Y_S8", "x", "X_S8"]   # theo thứ tự che_nang, che_nang, che_vua, khong_che
    audit.to_csv(tmp_path / "audit.csv", index=False)
    res = al.audit_score(tmp_path / "audit.csv")
    assert res.loc["tong", "n"] == 4
    assert res.loc["che_vua", "do_dung"] == 0.0              # tự động X_S8, tay "x" -> sai


# ---------------------------------------------------------------- hệ toạ độ WHAM

def test_wham_to_true_camera_keeps_pixel_and_scales_depth():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_wham_camera as cw

    Kh = {"fx": 2896.3, "fy": 2896.3, "cx": 1024.0, "cy": 1024.0}
    Kt = {"fx": 1497.2, "fy": 1496.7, "cx": 1024.6, "cy": 1051.4}
    P = np.array([[300.0, -200.0, 7000.0], [-150.0, 400.0, 6500.0]])
    Pt = cw.to_true_camera(P, Kh, Kt)
    np.testing.assert_allclose(mk.project(Pt, Kt), mk.project(P, Kh), atol=1e-6)
    np.testing.assert_allclose(Pt[:, 2], P[:, 2] * np.sqrt(Kt["fx"] * Kt["fy"]) / Kh["fx"])


def test_check_wham_camera_overlay_matches_by_frame_and_person(tmp_path):
    """Chế độ --meta: GT lấy từ gt_3d_scene.csv theo (frame, person), không theo frame nguồn."""
    import json

    import pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_wham_camera as cw

    rows = []
    for f, person, l, r in ((0, "X_S8", (0, 0, 3000), (200, 0, 3000)),
                            (0, "Y_S8", (0, 0, 6000), (100, 0, 6000)),
                            (1, "Y_S8", (50, 0, 6100), (150, 0, 6100))):
        rows.append({"frame": f, "person": person, "src_frame": 4750 + f,
                     **{f"left_hip_{a}": v for a, v in zip("xyz", l)},
                     **{f"right_hip_{a}": v for a, v in zip("xyz", r)}})
    pd.DataFrame(rows).to_csv(tmp_path / "gt_3d_scene.csv", index=False)
    K = {"fx": 1497.2, "fy": 1496.7, "cx": 1024.6, "cy": 1051.4}
    (tmp_path / "overlay_meta.json").write_text(json.dumps(
        {"size": [2048, 2048], "gt": {"scene_camera": {"person": "X_S8", **K}}}))
    gt, K_scene = cw.load_overlay_gt(tmp_path / "overlay_meta.json")
    assert K_scene == K
    np.testing.assert_allclose(gt[(0, "Y_S8")], [50, 0, 6000])

    pd.DataFrame({"frame": [0, 0, 1, 2], "body_id": [3, 7, 7, 7],
                  "person": ["X_S8", "Y_S8", "", "Y_S8"]}).to_csv(tmp_path / "map.csv", index=False)
    mapping = cw.load_mapping(tmp_path / "map.csv")
    assert mapping == {(0, 3): "X_S8", (0, 7): "Y_S8", (2, 7): "Y_S8"}   # dòng trống bị bỏ

    J = np.eye(3)                                    # 3 "khớp" = 3 đỉnh; hông SMPL = khớp 1, 2
    verts = lambda h: np.array([[0, 0, 0], [h[0] - .1, h[1], h[2]], [h[0] + .1, h[1], h[2]]])  # noqa: E731
    results = {3: {"frame_ids": [0], "verts": [verts((0.1, 0, 3.0))]},
               7: {"frame_ids": [0, 2], "verts": [verts((0.05, 0, 6.0)), verts((0, 0, 9.0))]}}
    W, G, rec = cw.matched_overlay(results, J, mapping, gt)
    assert [(r["frame"], r["person"]) for r in rec] == [(0, "X_S8"), (0, "Y_S8")]  # frame 2: không có GT
    np.testing.assert_allclose(W, [[100, 0, 3000], [50, 0, 6000]])
    np.testing.assert_allclose(G, [[100, 0, 3000], [50, 0, 6000]])

    pd.DataFrame({"frame": [0], "person": ["X_S8"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit, match="body_id"):
        cw.load_mapping(tmp_path / "bad.csv")
