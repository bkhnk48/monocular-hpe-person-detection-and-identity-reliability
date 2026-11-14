"""output_wham_video nhận Drive File ID: tải một lần vào out_dir khi ghép video.

Chạy:  python3 -m pytest -q tests/test_output_video.py
"""
import json

import pandas as pd
import pytest

from pose_assoc.build_network import mapping_module

MOD = mapping_module()
VIDEO_ID = "1alTVEp6vOHS0D2HkoGQcPBUisqBojqeK"


def manifest(tmp_path, video):
    (tmp_path / "local.mp4").write_bytes(b"x")
    m = {"base_dir": str(tmp_path), "smpl_model": "smpl.pkl", "output_dir": "out",
         "validation": [], "test": [{"name": "s", "wham_pkl": "w.pkl",
                                     "gt": {"type": "harmony4d", "path": "poses3d"},
                                     "output_wham_video": video, "fps": 20}]}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(m), encoding="utf-8")
    return MOD.load_manifest(path)["test"][0]


def test_drive_id_becomes_cache_path_in_out_dir(tmp_path):
    seq = manifest(tmp_path, VIDEO_ID)
    assert seq["output_wham_video_drive"] == VIDEO_ID
    assert seq["output_wham_video"] == seq["out_dir"] / f"output_wham_{VIDEO_ID}.mp4"


def test_local_path_and_empty_unchanged(tmp_path):
    seq = manifest(tmp_path, "local.mp4")
    assert seq["output_wham_video_drive"] is None
    assert seq["output_wham_video"] == (tmp_path / "local.mp4").resolve()
    assert manifest(tmp_path, "")["output_wham_video"] is None


def test_fetch_downloads_once_then_reuses(tmp_path, monkeypatch):
    calls = []

    def fake_download(file_id, dest, what="PKL"):
        calls.append((file_id, what))
        dest.write_bytes(b"video")
        return dest

    monkeypatch.setattr(MOD, "download_drive_file", fake_download)
    seq = manifest(tmp_path, VIDEO_ID)
    first = MOD.fetch_output_wham_video(seq)
    assert first.is_file() and calls == [(VIDEO_ID, "video WHAM")]
    assert MOD.fetch_output_wham_video(seq) == first and len(calls) == 1


def test_fetch_failure_falls_back_without_crash(tmp_path, monkeypatch, capsys):
    def broken(file_id, dest, what="PKL"):
        (dest.parent / (dest.name + "abc.part")).write_bytes(b"do dang")
        raise RuntimeError("không có mạng")

    monkeypatch.setattr(MOD, "download_drive_file", broken)
    seq = manifest(tmp_path, VIDEO_ID)
    video = MOD.fetch_output_wham_video(seq)
    assert not video.is_file()
    assert not list(video.parent.glob("*.part"))            # dọn tệp tải dở
    assert "không tải được output_wham_video" in capsys.readouterr().out


def test_local_video_is_never_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(MOD, "download_drive_file",
                        lambda *a, **k: pytest.fail("không được tải"))
    seq = manifest(tmp_path, "local.mp4")
    assert MOD.fetch_output_wham_video(seq) == (tmp_path / "local.mp4").resolve()


def test_combined_video_rebuilt_when_sources_change(tmp_path, monkeypatch):
    """combined_frames.mp4 cũ ghép với video khác (ví dụ trỏ nhầm cam07) phải được ghép lại."""
    built = []
    monkeypatch.setattr(MOD, "render_animation",
                        lambda poses, skel, n, fps, path, title: path.write_bytes(b"a"))
    monkeypatch.setattr(MOD, "combine_videos",
                        lambda videos, out, fps, n, has: (built.append(list(videos)),
                                                          out.write_bytes(b"c")))
    monkeypatch.setattr(MOD, "video_info", lambda path: (20.0, 1.0))
    monkeypatch.setattr(MOD, "wham_poses", lambda df: {})
    seq = manifest(tmp_path, "local.mp4")
    seq["out_dir"].mkdir(parents=True)
    df, gt = pd.DataFrame({"frame_id": [0, 1]}), {0: {}, 1: {}}
    MOD.build_videos(seq, df, gt, "coco17", False)
    MOD.build_videos(seq, df, gt, "coco17", False)          # cùng nguồn -> dùng lại
    assert len(built) == 1 and len(built[0]) == 3
    seq["output_wham_video"] = None                         # đổi nguồn -> ghép lại
    MOD.build_videos(seq, df, gt, "coco17", False)
    assert len(built) == 2 and len(built[1]) == 2


def test_status_shows_drive_video(tmp_path, capsys):
    seq = manifest(tmp_path, VIDEO_ID)
    MOD.print_status({"validation": [], "test": [seq]})
    assert "3 video (Drive)" in capsys.readouterr().out
