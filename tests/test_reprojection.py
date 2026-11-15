"""C_2D (beta): sai số 2D giữa detection WHAM và người GT trên ảnh, nguồn GT 2D
(annot2, CSV, poses2d của harmony4d, Drive) và chốt chặn trong tune.

Chạy:  python3 -m pytest -q tests/test_reprojection.py
"""
import io
import json
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from pose_assoc import tune
from pose_assoc.build_network import OVERRIDES, load_gt_2d, load_params, mapping_module
from pose_assoc.costs import (LIMBS, CostParams, person_scale_2d, reprojection_cost,
                              reprojection_error_2d)
from pose_assoc.evaluation import assignment_table, objective_breakdown
from pose_assoc.network import (apply_params, attach_gt_2d, build_network, has_gt_2d,
                                nodes_frame)
from pose_assoc.solvers.multi_mcf import solve_network

MOD = mapping_module()
RNG = np.random.default_rng(0)
TEMPLATE = RNG.uniform(0, 100, size=(12, 2))          # khung người 2D cỡ ~100 px
POSE = RNG.normal(scale=0.3, size=(12, 3))            # pose GT 3D (m)
OTHER = RNG.normal(scale=0.3, size=(12, 3))           # pose của một người khác
DRIVE_ID = "1bQx0Yd3vGfZPq8sZr7mT2kLwN4hJcAeU"


def at(x, y):
    return TEMPLATE + np.array([x, y], float)


# ------------------------------------------------------------------ công thức


def test_error_is_mean_joint_distance_over_gt_size():
    g = at(500, 500)
    assert reprojection_error_2d(g, g) == 0.0
    det = g + [6.0, 8.0]                               # mọi khớp lệch 10 px
    assert reprojection_error_2d(det, g) == pytest.approx(10.0 / person_scale_2d(g))
    big = g * 2.0                                      # chia cho cỡ người GT, không phải detection
    assert reprojection_error_2d(big + [10.0, 0.0], big) == pytest.approx(
        10.0 / person_scale_2d(big))


def test_error_nan_without_enough_joints_and_cost_zero():
    g = at(0, 0)
    assert np.isnan(reprojection_error_2d(None, g))
    assert np.isnan(reprojection_error_2d(g, None))
    det = g.copy()
    det[:5] = np.nan                                   # còn 7 khớp < 8
    assert np.isnan(reprojection_error_2d(det, g))
    p = CostParams(beta=2.0, reproj_scale=0.05)
    assert reprojection_cost([np.nan, 0.1], p) == pytest.approx([0.0, 4.0])
    assert reprojection_cost([0.1], CostParams()) == pytest.approx([0.0])   # beta = 0


# ------------------------------------------------------------------ mạng nhỏ


def switch_scene():
    """1 người GT ở x = 300 px, 5 frame. body 0 = đúng người (PA nhỏ, ngay vị trí GT 2D).
    body 1 = người khác ở x = 1500 px, riêng frame 2 có pose trùng hẳn GT (PA = 0),
    nên chỉ xét pose thì đường đi nhảy sang body 1 ở frame 2."""
    rng = np.random.default_rng(1)
    gt = {f: {"P": POSE} for f in range(5)}
    dets, det_uv, gt_uv = {}, {}, {}
    for f in range(5):
        near = POSE + rng.normal(scale=0.01, size=(12, 3))
        dets[f] = [(0, near), (1, POSE.copy() if f == 2 else OTHER)]
        det_uv[(f, 0)], det_uv[(f, 1)] = at(302, 400), at(1500, 400)
        gt_uv[(f, "P")] = at(300, 400)
    return dets, gt, det_uv, gt_uv


def chosen(net, p):
    apply_params(net, p)
    flow, info = solve_network(net)
    rows = assignment_table(net, flow).sort_values("frame")
    return [int(b) for b in rows["body_id"].fillna(-1)], flow, info   # -1 = missed


def test_beta_keeps_identity_that_pose_alone_switches():
    dets, gt, det_uv, gt_uv = switch_scene()
    p = CostParams(gamma=0.0)
    net = attach_gt_2d(build_network(dets, gt, p), det_uv, gt_uv)
    assert chosen(net, p)[0] == [0, 0, 1, 0, 0]
    assert chosen(net, CostParams(gamma=0.0, beta=0.2))[0] == [0, 0, 0, 0, 0]


def test_c2d_added_to_detection_nodes_only():
    dets, gt, det_uv, gt_uv = switch_scene()
    net = attach_gt_2d(build_network(dets, gt, CostParams()), det_uv, gt_uv)
    base = apply_params(net, CostParams()).node_costs.copy()
    p = CostParams(beta=0.5, reproj_scale=0.1)
    cost = apply_params(net, p).node_costs
    for n in net.nodes:
        if n["kind"] == "det":
            assert np.isfinite(n["e2d"])
            want = 0.5 * n["e2d"] / 0.1
            assert cost[n["id"]] - base[n["id"]] == pytest.approx(want)
        else:
            assert cost[n["id"]] == base[n["id"]]
    assert net.gt_2d_coverage == (10, 10)
    assert "e2d" in nodes_frame(net).columns


def test_beta_zero_leaves_costs_unchanged():
    dets, gt, det_uv, gt_uv = switch_scene()
    p = CostParams()
    net = build_network(dets, gt, p)
    before = net.node_costs.copy()
    apply_params(attach_gt_2d(net, det_uv, gt_uv), p)
    assert np.array_equal(net.node_costs, before)


def test_beta_without_gt_2d_is_skipped_not_crashing():
    """build_network() tự gọi apply_params() ở cuối, trước attach_gt_2d()."""
    dets, gt, _, _ = switch_scene()
    plain = build_network(dets, gt, CostParams())
    net = build_network(dets, gt, CostParams(beta=1.0))
    assert not has_gt_2d(net)
    assert np.array_equal(net.node_costs, plain.node_costs)
    assert "e2d" not in nodes_frame(net).columns


def test_attach_refuses_gt_2d_that_matches_nothing():
    dets, gt, det_uv, gt_uv = switch_scene()
    wrong_names = {(f, "aria01"): uv for (f, _), uv in gt_uv.items()}
    with pytest.raises(ValueError, match="tên người"):
        attach_gt_2d(build_network(dets, gt, CostParams()), det_uv, wrong_names)


def test_objective_breakdown_splits_reprojection():
    dets, gt, det_uv, gt_uv = switch_scene()
    p = CostParams(gamma=0.5, beta=0.3)
    net = attach_gt_2d(build_network(dets, gt, p), det_uv, gt_uv)
    _, flow, info = chosen(net, p)
    o = objective_breakdown(net, flow, p)
    assert o["reprojection"] > 0
    parts = ("pose", "reprojection", "false_positive", "false_negative",
             "temporal_and_gaps", "displacement", "boundary")
    assert sum(o[k] for k in parts) == pytest.approx(info["objective"])


# ------------------------------------------------------------------ nguồn GT 2D


def manifest(tmp_path, name, gt):
    m = {"base_dir": str(tmp_path), "smpl_model": "smpl.pkl", "output_dir": "out",
         "validation": [], "test": [{"name": name, "wham_pkl": "w.pkl", "gt": gt, "fps": 20}]}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(m), encoding="utf-8")
    return MOD.load_manifest(path)["test"][0]


def write_csv(path, frames=(0, 1), persons=("X_S8", "Y_S8")):
    cols = ["frame", "person", "src_frame"] + [f"{j}_{a}" for j in LIMBS for a in "uv"]
    rows = [",".join(cols)]
    for f in frames:
        for i, p in enumerate(persons):
            uv = at(100 * i + f, 50).reshape(-1)
            rows.append(",".join([str(f), p, "7"] + [f"{v:.3f}" for v in uv]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def h4d_npy(value: float) -> np.ndarray:
    pts = np.zeros((45, 2))
    pts[:, 0] = np.arange(45)                          # u = chỉ số khớp SMPL
    pts[:, 1] = value
    return pts


def write_poses2d(root, cams=("cam01", "cam07"), n_frames=2):
    for cam in cams:
        (root / cam).mkdir(parents=True)
        for i in range(n_frames):
            np.save(root / cam / f"{i + 1:05d}.npy", {"aria01": h4d_npy(i), "aria02": h4d_npy(9)},
                    allow_pickle=True)
    return root


def test_csv_source_from_path_2d(tmp_path):
    write_csv(tmp_path / "gt_2d.csv")
    seq = manifest(tmp_path, "ovl", {"type": "mpi_overlay_csv", "path": "gt.csv",
                                     "path_2d": "gt_2d.csv"})
    assert seq["gt"]["path_2d"] == (tmp_path / "gt_2d.csv").resolve()
    src = MOD.resolve_gt_2d(seq)
    assert src["kind"] == "csv"
    gt2d = MOD.load_gt_2d(src)
    assert set(gt2d) == {0, 1} and set(gt2d[1]) == {"X_S8", "Y_S8"}
    assert gt2d[1]["Y_S8"] == pytest.approx(at(101, 50), abs=1e-3)
    flat = load_gt_2d(MOD, seq)
    assert set(flat) == {(0, "X_S8"), (0, "Y_S8"), (1, "X_S8"), (1, "Y_S8")}


def test_csv_missing_columns_is_an_error(tmp_path):
    (tmp_path / "bad.csv").write_text("frame,person,left_shoulder_u\n0,A,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu cột"):
        MOD.load_gt_2d({"kind": "csv", "path": tmp_path / "bad.csv"})


def test_harmony4d_poses2d_picks_camera_from_name(tmp_path):
    write_poses2d(tmp_path / "poses2d")
    seq = manifest(tmp_path, "h4d_041_grappling2_cam07",
                   {"type": "harmony4d", "path": "poses3d", "path_2d": "poses2d"})
    src = MOD.resolve_gt_2d(seq)
    assert src["kind"] == "npy" and src["path"].name == "cam07"
    gt2d = MOD.load_gt_2d(src)
    assert set(gt2d) == {0, 1} and set(gt2d[0]) == {"aria01", "aria02"}
    # 12 khớp LIMBS lấy theo chỉ số SMPL-24 của WHAM (left_shoulder = 16, ...)
    want = [MOD.WHAM_SMPL_IDX[j] for j in LIMBS]
    assert gt2d[1]["aria01"][:, 0] == pytest.approx(want)
    assert gt2d[1]["aria01"][:, 1] == pytest.approx(np.ones(12))


def test_harmony4d_frames_follow_file_names_not_order(tmp_path, capsys):
    """Tải lên Drive bị đứt: thiếu tệp và có bản nhân '00002(1).npy'. Frame phải theo số
    trong tên (00001.npy = frame 0), không theo thứ tự tệp, để các frame sau không bị lệch."""
    cam = tmp_path / "poses2d" / "cam07"
    write_poses2d(tmp_path / "poses2d", cams=("cam07",), n_frames=5)
    (cam / "00003.npy").unlink()
    np.save(cam / "00002(1).npy", {"aria01": h4d_npy(99)}, allow_pickle=True)
    gt2d = MOD.load_gt_2d({"kind": "npy", "path": cam})
    out = capsys.readouterr().out
    assert sorted(gt2d) == [0, 1, 3, 4]
    assert gt2d[3]["aria01"][:, 1] == pytest.approx(np.full(12, 3.0))   # tệp 00004.npy
    assert "bỏ qua 1 tệp tên lạ" in out and "thiếu 1 frame" in out


def test_poses3d_frames_follow_file_names_too(tmp_path):
    d = tmp_path / "poses3d"
    d.mkdir()
    for i in (1, 2, 4):
        np.save(d / f"{i:05d}.npy", {"aria01": np.full((17, 4), float(i))}, allow_pickle=True)
    poses = MOD.h4d_gt_poses(d)
    assert sorted(poses) == [0, 1, 3] and poses[3]["aria01"][0, 0] == 4.0


def test_zip_missing_the_camera_is_a_clear_error(tmp_path, monkeypatch):
    """Như zip poses2d của 041 trên Drive: chỉ có vài camera, không có camera cần dùng."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for cam in ("cam02", "cam05"):
            b = io.BytesIO()
            np.save(b, {"aria01": h4d_npy(0)}, allow_pickle=True)
            zf.writestr(f"poses2d/{cam}/00001.npy", b.getvalue())
    fake_drive(monkeypatch, buf.getvalue())
    seq = manifest(tmp_path, "h4d_041_grappling2_cam01",
                   {"type": "harmony4d", "path": DRIVE_ID, "path_2d": DRIVE_ID})
    with pytest.raises(ValueError, match="không có thư mục cam01.*cam02, cam05"):
        MOD.resolve_gt_2d(seq)
    assert not list(seq["out_dir"].glob("gt2d_*"))    # không để lại bản giải nén dở


def test_harmony4d_camera_dir_must_match(tmp_path):
    write_poses2d(tmp_path / "poses2d", cams=("cam01",))
    seq = manifest(tmp_path, "h4d_041_grappling2_cam07",
                   {"type": "harmony4d", "path": "poses3d", "path_2d": "poses2d/cam01"})
    with pytest.raises(ValueError, match="cam01"):
        MOD.resolve_gt_2d(seq)


def test_mpi_uses_annot2_of_same_camera(tmp_path):
    F = 3
    cams = np.empty((2, 1), dtype=object)
    for c in range(2):
        a = np.zeros((F, 28, 2))
        a[..., 0] = np.arange(28) + 100 * c            # u = chỉ số khớp MPI (+100 ở camera 1)
        a[..., 1] = np.arange(F)[:, None]
        cams[c, 0] = a.reshape(F, 56)
    savemat(tmp_path / "annot.mat", {"annot2": cams})
    seq = manifest(tmp_path, "mpi_S8_Seq1_cam1",
                   {"type": "mpi", "path": "annot.mat", "camera": 1, "person": "S8"})
    src = MOD.resolve_gt_2d(seq)
    assert src["kind"] == "mpi"
    gt2d = MOD.load_gt_2d(src)
    assert set(gt2d) == {0, 1, 2} and set(gt2d[2]) == {"S8"}
    want = [MOD.MPI_IDX[j] + 100 for j in LIMBS]
    assert gt2d[2]["S8"][:, 0] == pytest.approx(want)
    assert gt2d[2]["S8"][:, 1] == pytest.approx(np.full(12, 2.0))


def test_mpi_overlay_reads_gt_2d_next_to_meta(tmp_path):
    (tmp_path / "ovl").mkdir()
    (tmp_path / "ovl" / "overlay_meta.json").write_text("{}", encoding="utf-8")
    seq = manifest(tmp_path, "ovl", {"type": "mpi_overlay", "path": "ovl/overlay_meta.json"})
    assert MOD.resolve_gt_2d(seq) is None              # chưa có gt_2d.csv
    write_csv(tmp_path / "ovl" / "gt_2d.csv")
    assert MOD.resolve_gt_2d(seq)["path"] == tmp_path / "ovl" / "gt_2d.csv"


def test_no_source_is_explicit_error(tmp_path):
    seq = manifest(tmp_path, "h4d_041_grappling2_cam07", {"type": "harmony4d", "path": "poses3d"})
    assert MOD.resolve_gt_2d(seq) is None
    with pytest.raises(ValueError, match="path_2d"):
        load_gt_2d(MOD, seq)


def fake_drive(monkeypatch, payload: bytes):
    calls = []

    def download(file_id, dest, what="PKL"):
        calls.append((file_id, what))
        dest.write_bytes(payload)
        return dest

    monkeypatch.setattr(MOD, "download_drive_file", download)
    return calls


def test_drive_csv_downloaded_once(tmp_path, monkeypatch):
    calls = fake_drive(monkeypatch, write_csv(tmp_path / "src.csv").read_bytes())
    seq = manifest(tmp_path, "ovl", {"type": "mpi_overlay_csv", "path": DRIVE_ID,
                                     "path_2d": DRIVE_ID})
    assert seq["gt"]["path_2d_is_drive"]
    src = MOD.resolve_gt_2d(seq)
    assert src == {"kind": "csv", "path": seq["out_dir"] / f"gt2d_{DRIVE_ID}.csv"}
    assert MOD.resolve_gt_2d(seq) == src and len(calls) == 1
    assert not list(seq["out_dir"].glob("*.download"))


def test_drive_zip_of_poses2d_extracted_once(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for cam in ("cam01", "cam07"):
            for i in range(2):
                b = io.BytesIO()
                np.save(b, {"aria01": h4d_npy(i)}, allow_pickle=True)
                zf.writestr(f"poses2d/{cam}/{i + 1:05d}.npy", b.getvalue())
    calls = fake_drive(monkeypatch, buf.getvalue())
    seq = manifest(tmp_path, "h4d_002_hugging_cam07",
                   {"type": "harmony4d", "path": DRIVE_ID, "path_2d": DRIVE_ID})
    src = MOD.resolve_gt_2d(seq)
    cache = seq["out_dir"] / f"gt2d_{DRIVE_ID}"
    assert src["kind"] == "npy" and src["path"].name == "cam07"
    assert src["path"].is_relative_to(cache)
    assert MOD.resolve_gt_2d(seq) == src and len(calls) == 1
    assert set(MOD.load_gt_2d(src)) == {0, 1}
    # zip của cả scene: chỉ giải nén camera của sequence
    assert {f.parent.name for f in cache.rglob("*.npy")} == {"cam07"}


def test_scene_zip_made_like_colab_notebook(tmp_path):
    """Zip do scripts/harmony4d_drive_index_colab.ipynb tạo (shutil.make_archive với
    base_dir="poses2d"), ghi thẳng làm gt.path_2d dạng tệp .zip trên máy."""
    import shutil
    write_poses2d(tmp_path / "work" / "poses2d", cams=("cam01", "cam07", "cam12"))
    archive = shutil.make_archive(str(tmp_path / "poses2d"), "zip",
                                  root_dir=tmp_path / "work", base_dir="poses2d")
    for cam in ("cam07", "cam12"):
        seq = manifest(tmp_path, f"h4d_002_hugging_{cam}",
                       {"type": "harmony4d", "path": "poses3d", "path_2d": "poses2d.zip"})
        src = MOD.resolve_gt_2d(seq)
        assert src["kind"] == "npy" and src["path"].name == cam
        assert {f.parent.name for f in (seq["out_dir"] / "gt2d_poses2d").rglob("*.npy")} == {cam}
        assert MOD.load_gt_2d(src)[1]["aria02"][:, 1] == pytest.approx(np.full(12, 9.0))
    assert archive.endswith("poses2d.zip")


# ------------------------------------------------------------------ tune và CLI


def test_require_gt_2d_blocks_beta_sweep_on_networks_without_it():
    dets, gt, det_uv, gt_uv = switch_scene()
    with_2d = attach_gt_2d(build_network(dets, gt, CostParams()), det_uv, gt_uv)
    without = build_network(dets, gt, CostParams())
    data = [{"name": "co", "net": with_2d}, {"name": "thieu", "net": without}]
    tune.require_gt_2d(data, {"alpha": [1.0]}, CostParams())          # không quét beta
    tune.require_gt_2d(data, {"beta": [0.0]}, CostParams())
    with pytest.raises(SystemExit, match="--exclude thieu"):
        tune.require_gt_2d(data, {"beta": [0.0, 0.5]}, CostParams())
    with pytest.raises(SystemExit, match="thieu"):
        tune.require_gt_2d(data, {"alpha": [1.0]}, CostParams(beta=0.2))
    tune.require_gt_2d(data[:1], {"beta": [0.5]}, CostParams())


def test_gt_signature_tracks_path_2d(tmp_path):
    write_csv(tmp_path / "gt_2d.csv")
    (tmp_path / "gt.csv").write_text("frame,person\n0,X\n", encoding="utf-8")
    plain = manifest(tmp_path, "s", {"type": "mpi_overlay_csv", "path": "gt.csv"})
    with_2d = manifest(tmp_path, "s", {"type": "mpi_overlay_csv", "path": "gt.csv",
                                       "path_2d": "gt_2d.csv"})
    assert tune.gt_signature(plain) != tune.gt_signature(with_2d)
    assert tune.gt_signature(with_2d).startswith(tune.gt_signature(plain) + "|2d:")


def test_cli_and_round_file_know_beta():
    assert OVERRIDES["beta"] == "beta"
    args = tune.parse_args(["--beta", "0.3", "--grid", "beta=0,0.5"])
    assert load_params(args.config, args).beta == 0.3
    assert tune.parse_grid(args.grid)["beta"] == [0.0, 0.5]
    p = CostParams(alpha=1.5, eta=0.1, beta=0.25, reproj_scale=0.1)
    assert CostParams.from_config(tune.association_block(p)) == p
