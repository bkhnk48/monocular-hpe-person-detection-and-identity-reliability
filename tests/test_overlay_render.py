"""Test CLI và quy trình generate -> sửa tay -> render của scripts/make_overlay_pair.py.

    /usr/bin/python3 -m pytest -q tests/test_overlay_render.py

Không cần video thật, annot.mat thật hay FFmpeg thật: nguồn video và ffmpeg được thay bằng
đối tượng giả (open_source, ffmpeg_writer, export_gt bị monkeypatch).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_overlay_pair as mk  # noqa: E402
from test_overlay import standing_skeleton  # noqa: E402


class _FakeStdin:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, data):
        self.frames.append(data)

    def close(self):
        self.closed = True


class _FakeWriter:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        return 0


def _write_png(path: Path, image: np.ndarray):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_render_frames_prefers_edited_pngs(tmp_path, monkeypatch):
    """Ưu tiên frames_edit; frame chưa sửa phải lấy từ frames_raw."""
    raw = tmp_path / "frames_raw"
    edit = tmp_path / "frames_edit"

    originals = [
        np.full((20, 30, 3), 10, dtype=np.uint8),
        np.full((20, 30, 3), 20, dtype=np.uint8),
        np.full((20, 30, 3), 30, dtype=np.uint8),
    ]
    for i, image in enumerate(originals):
        _write_png(raw / f"{i:06d}.png", image)

    edited = np.full((20, 30, 3), 99, dtype=np.uint8)
    _write_png(edit / "000001.png", edited)

    fake_writer = _FakeWriter()
    monkeypatch.setattr(
        mk, "ffmpeg_writer", lambda *args, **kwargs: fake_writer
    )

    meta = {
        "n_frames": 3,
        "fps": 25.0,
        "size": [30, 20],
        "out_scale": 1.0,
    }
    assert mk.render_frames(tmp_path, meta) == [1]

    assert len(fake_writer.stdin.frames) == 3
    rendered = [
        np.frombuffer(data, dtype=np.uint8).reshape(20, 30, 3)
        for data in fake_writer.stdin.frames
    ]

    np.testing.assert_array_equal(rendered[0], originals[0])
    np.testing.assert_array_equal(rendered[1], edited)
    np.testing.assert_array_equal(rendered[2], originals[2])
    assert fake_writer.stdin.closed
    assert fake_writer.wait_called


def test_render_frames_fails_when_frame_missing(tmp_path):
    """Báo lỗi nếu một frame không tồn tại ở cả frames_raw và frames_edit."""
    raw = tmp_path / "frames_raw"
    raw.mkdir()
    _write_png(raw / "000000.png", np.zeros((20, 30, 3), dtype=np.uint8))

    meta = {
        "n_frames": 2,
        "fps": 25.0,
        "size": [30, 20],
        "out_scale": 1.0,
    }

    with pytest.raises(SystemExit, match="Thiếu"):
        mk.render_frames(tmp_path, meta)


def test_render_frames_rejects_wrong_dimensions(tmp_path, monkeypatch):
    """Không cho render nếu PNG chỉnh sửa có kích thước khác video đầu ra."""
    raw = tmp_path / "frames_raw"
    edit = tmp_path / "frames_edit"
    _write_png(raw / "000000.png", np.zeros((20, 30, 3), dtype=np.uint8))
    _write_png(edit / "000000.png", np.zeros((20, 40, 3), dtype=np.uint8))

    fake_writer = _FakeWriter()
    monkeypatch.setattr(
        mk, "ffmpeg_writer", lambda *args, **kwargs: fake_writer
    )

    meta = {
        "n_frames": 1,
        "fps": 25.0,
        "size": [30, 20],
        "out_scale": 1.0,
    }

    with pytest.raises(SystemExit, match="Kích thước"):
        mk.render_frames(tmp_path, meta)

    assert fake_writer.stdin.closed
    assert fake_writer.wait_called


def test_render_frames_warns_about_ignored_edits(tmp_path, monkeypatch, capsys):
    """Frame sửa lưu .jpg hoặc sai tên thì không được dùng: phải báo, không im lặng."""
    raw, edit = tmp_path / "frames_raw", tmp_path / "frames_edit"
    for i in range(2):
        _write_png(raw / f"{i:06d}.png", np.zeros((20, 30, 3), np.uint8))
    _write_png(edit / "000001.jpg", np.full((20, 30, 3), 99, np.uint8))
    _write_png(edit / "frame5.png", np.full((20, 30, 3), 99, np.uint8))
    monkeypatch.setattr(mk, "ffmpeg_writer", lambda *a, **k: _FakeWriter())
    edited = mk.render_frames(tmp_path, {"n_frames": 2, "fps": 25.0, "size": [30, 20]})
    assert edited == []
    out = capsys.readouterr().out
    assert "Bỏ qua 2 tệp" in out and "000001.jpg" in out and "frame5.png" in out


# ---------------------------------------------------------------- tham số dòng lệnh

@pytest.mark.parametrize("spec", ["S8:0", "S8:0:4750", "5:5", "10:3", "-1:10", "0:10:20", "a:b"])
def test_parse_segment_rejects_bad_ranges(spec):
    with pytest.raises(SystemExit):
        mk.parse_segment(spec)


def test_parse_segment_reads_start_end():
    assert mk.parse_segment("4750:6250") == {"start": 4750, "end": 6250}


def test_video_identity_reads_subject_seq_camera(tmp_path):
    v = tmp_path / "S8" / "Seq1" / "imageSequences" / "video_cam4.avi"
    assert {k: mk.video_identity(v)[k] for k in ("subject", "seq", "camera")} == \
        {"subject": "S8", "seq": "Seq1", "camera": 4}
    assert mk.video_identity(v, "S1", 2)["subject"] == "S1"
    assert mk.video_identity(v, "S1", 2)["camera"] == 2
    with pytest.raises(SystemExit, match="subject/camera"):
        mk.video_identity(tmp_path / "clip.avi")


def test_annot_path_refuses_seq2_video_with_seq1_pattern(tmp_path):
    src = {"subject": "S8", "camera": 0, "video": tmp_path / "S8/Seq2/imageSequences/video_cam0.avi",
           "seq": "Seq2"}
    with pytest.raises(SystemExit, match="Seq2"):
        mk.annot_path(src, mk.ANNOT_PATTERN)              # mẫu mặc định chỉ có annot Seq1
    assert "S8/Seq2/annot.mat" in str(mk.annot_path(src, "data/{subject}/{seq}/annot.mat"))
    assert mk.annot_path({**src, "seq": "Seq1"}, mk.ANNOT_PATTERN).name == "annot.mat"
    with pytest.raises(SystemExit, match="SeqN"):
        mk.annot_path({**src, "seq": None}, "data/{subject}/{seq}/annot.mat")


def test_check_overlap_refuses_same_timeline_unless_allowed(tmp_path, capsys):
    a, b = tmp_path / "S8.mat", tmp_path / "S1.mat"
    X = {"annot": a, "start": 0}
    with pytest.raises(SystemExit, match="allow-overlap"):
        mk.check_overlap(X, {"annot": a, "start": 500}, 1500, 100)      # giao nhau 1000 frame
    with pytest.raises(SystemExit):
        mk.check_overlap(X, {"annot": a, "start": 1550}, 1500, 100)     # cách nhau 50 < 100
    mk.check_overlap(X, {"annot": a, "start": 1600}, 1500, 100)         # cách đúng 100: được
    mk.check_overlap(X, {"annot": b, "start": 0}, 1500, 100)            # khác lần quay: được
    mk.check_overlap(X, {"annot": a, "start": 0}, 1500, 100, allow=True)
    assert "--allow-overlap" in capsys.readouterr().out


# ---------------------------------------------------------------- generate -> sửa -> render

class _FakeCap:
    def __init__(self, img):
        self.img = img

    def read(self):
        return True, self.img.copy()


def _fake_sources(tmp_path, monkeypatch, F=400):
    """open_source giả: ảnh 200x200 phông xanh, một người xám đứng giữa; cùng annot.mat."""
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (40, 140, 40)
    img[25:185, 85:116] = (200, 200, 200)
    skel = standing_skeleton(100, top=25, h=160)

    def fake_open(src, args):
        src.update(annot=tmp_path / "S8" / "annot.mat", j2d=np.repeat(skel[None], F, 0),
                   body_px=np.full(F, np.nan), n=F, size=(200, 200), fps=25.0, cap=_FakeCap(img))
        return src

    monkeypatch.setattr(mk, "open_source", fake_open)
    monkeypatch.setattr(mk, "export_gt", lambda out: {})      # GT cần annot.mat thật
    monkeypatch.setattr(mk, "print_gt", lambda out, gt: None)
    return tmp_path / "S8" / "Seq1" / "imageSequences" / "video_cam0.avi"


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["make_overlay_pair.py", *argv])
    mk.main()


def test_generate_edit_render_round_trip(tmp_path, monkeypatch, capsys):
    video = _fake_sources(tmp_path, monkeypatch)
    root = tmp_path / "out"
    common = ["--x-video", str(video), "--out-root", str(root), "--preview-every", "0"]
    # --y 200:400 nhưng --frames 6: tên thư mục và meta ghi đoạn thực sự dùng (200:206)
    _run(monkeypatch, "--x", "0:6", "--y", "200:400", "--frames", "6", "--mode", "generate", *common)
    out = root / "ovl_S8c0f0-6_S8c0f200-206_s0.5"
    assert sorted(p.name for p in (out / "frames_raw").iterdir()) == [f"{i:06d}.png" for i in range(6)]
    assert not (out / "composite.mp4").exists()
    meta = json.loads((out / "overlay_meta.json").read_text())
    assert meta["video"] is None and meta["frames_dir"] == "frames_raw"
    y = meta["persons"]["Y_S8"]
    assert (y["start"], y["end"]) == (200, 206)
    assert not Path(y["annot"]).is_absolute()                        # đường dẫn tương đối
    assert (out / y["annot"]).resolve() == (tmp_path / "S8" / "annot.mat").resolve()

    import cv2
    edited = cv2.imread(str(out / "frames_raw" / "000003.png"))
    edited[:20, :20] = (0, 0, 255)
    _write_png(out / "frames_edit" / "000003.png", edited)
    writer = _FakeWriter()
    monkeypatch.setattr(mk, "ffmpeg_writer", lambda *a, **k: writer)
    _run(monkeypatch, "--mode", "render", "--name", out.name, "--out-root", str(root))
    frames = [np.frombuffer(b, np.uint8).reshape(200, 200, 3) for b in writer.stdin.frames]
    assert len(frames) == 6
    np.testing.assert_array_equal(frames[3], edited)
    np.testing.assert_array_equal(frames[2], cv2.imread(str(out / "frames_raw" / "000002.png")))
    meta = json.loads((out / "overlay_meta.json").read_text())
    assert meta["video"] == "composite.mp4" and meta["edited_frames"] == [3]


def test_mode_all_writes_mp4_without_pngs(tmp_path, monkeypatch):
    video = _fake_sources(tmp_path, monkeypatch)
    writer = _FakeWriter()
    monkeypatch.setattr(mk, "ffmpeg_writer", lambda *a, **k: writer)
    _run(monkeypatch, "--x", "0:5", "--y", "200:205", "--x-video", str(video),
         "--out-root", str(tmp_path / "out"), "--preview-every", "0")
    out = tmp_path / "out" / "ovl_S8c0f0-5_S8c0f200-205_s0.5"
    assert len(writer.stdin.frames) == 5 and writer.stdin.closed
    assert not (out / "frames_raw").exists()                         # không tốn ~3 MB/frame
    assert json.loads((out / "overlay_meta.json").read_text())["video"] == "composite.mp4"


def test_same_video_overlapping_segments_are_refused(tmp_path, monkeypatch):
    """--x 0:1500 --y 0:2000 cùng video: Y chính là X thu nhỏ, cùng động tác -> dừng."""
    video = _fake_sources(tmp_path, monkeypatch, F=2100)
    with pytest.raises(SystemExit, match="allow-overlap"):
        _run(monkeypatch, "--x", "0:1500", "--y", "0:2000", "--x-video", str(video),
             "--frames", "1500", "--mode", "generate", "--out-root", str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()
