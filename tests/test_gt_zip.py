"""GT harmony4d dạng poses3d.zip: Drive File ID hoặc tệp .zip trên máy được giải nén
một lần vào out_dir; thư mục poses3d trên máy vẫn đọc như cũ.

Chạy:  python3 -m pytest -q tests/test_gt_zip.py
"""
import io
import json
import zipfile

import numpy as np
import pytest

from pose_assoc.build_network import mapping_module

MOD = mapping_module()
GT_ID = "1bQx0Yd3vGfZPq8sZr7mT2kLwN4hJcAeU"


def npy_bytes(value: float) -> bytes:
    buf = io.BytesIO()
    np.save(buf, {"aria01": np.full((17, 4), value)}, allow_pickle=True)
    return buf.getvalue()


def make_zip(path, n_frames=2, prefix="poses3d/", junk=False):
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n_frames):
            zf.writestr(f"{prefix}{i + 1:05d}.npy", npy_bytes(i))
        if junk:                                    # zip nén bằng Finder trên macOS
            zf.writestr(f"__MACOSX/{prefix}._00001.npy", b"rac")
    return path


def manifest(tmp_path, gt_path, gt_type="harmony4d"):
    m = {"base_dir": str(tmp_path), "smpl_model": "smpl.pkl", "output_dir": "out",
         "validation": [], "test": [{"name": "s", "wham_pkl": "w.pkl",
                                     "gt": {"type": gt_type, "path": gt_path}, "fps": 20}]}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(m), encoding="utf-8")
    return MOD.load_manifest(path)["test"][0]


def fake_drive(tmp_path, monkeypatch, n_frames=2):
    """download_drive_file giả: chép một zip có sẵn, ghi lại các lần tải."""
    calls, source = [], make_zip(tmp_path / "drive_copy.zip", n_frames)

    def download(file_id, dest, what="PKL"):
        calls.append((file_id, what))
        dest.write_bytes(source.read_bytes())
        return dest

    monkeypatch.setattr(MOD, "download_drive_file", download)
    return calls


def test_manifest_keeps_drive_id_for_harmony4d(tmp_path):
    seq = manifest(tmp_path, GT_ID)
    assert seq["gt"]["path_is_drive"] is True and seq["gt"]["path"] == GT_ID


def test_manifest_still_rejects_drive_id_for_mpi(tmp_path):
    with pytest.raises(ValueError, match="harmony4d"):
        manifest(tmp_path, GT_ID, gt_type="mpi")


def test_local_poses3d_dir_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "download_drive_file",
                        lambda *a, **k: pytest.fail("không được tải"))
    (tmp_path / "poses3d").mkdir()
    (tmp_path / "poses3d" / "00001.npy").write_bytes(npy_bytes(0))
    seq = manifest(tmp_path, "poses3d")
    gt = MOD.resolve_gt_path(seq)
    assert gt["path"] == (tmp_path / "poses3d").resolve()
    assert not seq["out_dir"].exists()                     # không tạo cache
    assert len(MOD.load_gt(gt)[0]) == 1


def test_drive_zip_downloaded_once_then_read(tmp_path, monkeypatch):
    calls = fake_drive(tmp_path, monkeypatch)
    seq = manifest(tmp_path, GT_ID)
    gt = MOD.resolve_gt_path(seq)
    assert gt["path"] == seq["out_dir"] / f"gt_{GT_ID}" / "poses3d"
    poses, skel = MOD.load_gt(gt)
    assert skel == "coco17" and sorted(poses) == [0, 1]
    assert poses[1]["aria01"].shape == (17, 3) and poses[1]["aria01"][0, 0] == 1
    assert not (seq["out_dir"] / f"gt_{GT_ID}.zip").exists()   # zip tải về đã xoá
    assert MOD.resolve_gt_path(seq)["path"] == gt["path"]      # lần 2: dùng lại
    assert calls == [(GT_ID, "GT poses3d.zip")]
    assert seq["gt"]["path"] == GT_ID                          # seq giữ nguyên ID


def test_local_zip_extracted_once_and_again_when_changed(tmp_path):
    zip_path = make_zip(tmp_path / "poses3d.zip", n_frames=2)
    seq = manifest(tmp_path, "poses3d.zip")
    gt = MOD.resolve_gt_path(seq)
    assert gt["path"] == seq["out_dir"] / "gt_poses3d" / "poses3d"
    assert len(MOD.load_gt(gt)[0]) == 2
    marker = seq["out_dir"] / "gt_poses3d" / MOD.H4D_GT_MARKER
    stamp = marker.stat().st_mtime_ns
    MOD.resolve_gt_path(seq)
    assert marker.stat().st_mtime_ns == stamp                  # không giải nén lại
    make_zip(zip_path, n_frames=3)                             # zip đổi -> giải nén lại
    assert len(MOD.load_gt(MOD.resolve_gt_path(seq))[0]) == 3


def test_zip_with_npy_at_root_and_macos_junk(tmp_path):
    make_zip(tmp_path / "gt.zip", n_frames=2, prefix="", junk=True)
    seq = manifest(tmp_path, "gt.zip")
    gt = MOD.resolve_gt_path(seq)
    assert gt["path"] == seq["out_dir"] / "gt_gt"
    assert not (gt["path"] / "__MACOSX").exists()
    assert len(MOD.load_gt(gt)[0]) == 2


@pytest.mark.parametrize("content, error", [(b"khong phai zip", "zip hợp lệ"),
                                            (None, "Không có tệp .npy")])
def test_bad_download_leaves_no_cache(tmp_path, monkeypatch, content, error):
    def download(file_id, dest, what="PKL"):
        if content is None:                                    # zip hợp lệ nhưng rỗng
            zipfile.ZipFile(dest, "w").close()
        else:
            dest.write_bytes(content)
        (dest.parent / (dest.name + "abc.part")).write_bytes(b"do dang")
        return dest

    monkeypatch.setattr(MOD, "download_drive_file", download)
    seq = manifest(tmp_path, GT_ID)
    with pytest.raises(ValueError, match=error):
        MOD.resolve_gt_path(seq)
    assert list(seq["out_dir"].iterdir()) == []                # không sót zip, .tmp, .part


def test_status_not_missing_for_zip_sources(tmp_path):
    make_zip(tmp_path / "poses3d.zip")
    assert MOD.sequence_status(manifest(tmp_path, "poses3d.zip")) != "THIẾU GT"
    assert MOD.sequence_status(manifest(tmp_path, GT_ID)) != "THIẾU GT"
    assert MOD.sequence_status(manifest(tmp_path, "khong_co.zip")) == "THIẾU GT"
