"""scripts/update_splits.py: đặt trường cho mọi h4d_<X>_camY, giữ nguyên phần còn lại của tệp.

Chạy:  python3 -m pytest -q tests/test_update_splits.py
"""
import difflib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("update_splits", ROOT / "scripts" / "update_splits.py")
US = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(US)


def entry(name, **extra):
    return {"name": name, "wham_pkl": "", "gt": {"type": "harmony4d", "path": "GT"}, **extra}


def write_splits(tmp_path):
    hand = '{\n  "_ghi_chu": "x",\n\n  "validation": [\n    {"name": "mpi_a", "image_size": [2048, 2048]}\n  ],\n'
    body = json.dumps([entry("h4d_009_ballroom2_cam01"), entry("h4d_009_ballroom2_cam12"),
                       entry("h4d_0090_ballroom2_cam01"), entry("h4d_041_grappling2_cam07")],
                      indent=2).replace("\n", "\n  ")
    path = tmp_path / "splits.json"
    path.write_text(hand + f'  "test": {body}\n}}\n', encoding="utf-8")
    return path


def run(path, *extra):
    return US.main(["--action", "009_ballroom2", "--set", "gt.path_2d=P2D",
                    "--set", "colmap_folder_ID=COL", "--splits", str(path), *extra])


def test_updates_only_matching_cams_and_keeps_other_text(tmp_path):
    path = write_splits(tmp_path)
    before = path.read_text(encoding="utf-8")
    assert run(path) == 0
    after = path.read_text(encoding="utf-8")
    data = json.loads(after)
    by = {e["name"]: e for e in data["test"]}
    for cam in ("01", "12"):
        e = by[f"h4d_009_ballroom2_cam{cam}"]
        assert e["gt"] == {"type": "harmony4d", "path": "GT", "path_2d": "P2D"}
        assert e["colmap_folder_ID"] == "COL"
    assert "path_2d" not in by["h4d_0090_ballroom2_cam01"]["gt"]       # không khớp nhầm
    assert "colmap_folder_ID" not in by["h4d_041_grappling2_cam07"]
    assert after.startswith(before[:before.index('"test"')])           # phần định dạng tay giữ nguyên
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    removed = [d[2:].strip() for d in diff if d.startswith("- ")]
    added = [d[2:].strip() for d in diff if d.startswith("+ ")]
    # mỗi cam khớp: dòng "path" và dấu "}" của gt thêm dấu phẩy, thêm 2 dòng mới
    assert removed == ['"path": "GT"', "}"] * 2
    assert sorted(added) == sorted(['"path": "GT",', '"path_2d": "P2D"', "},",
                                    '"colmap_folder_ID": "COL"'] * 2)


def test_second_run_and_dry_run_do_not_write(tmp_path, capsys):
    path = write_splits(tmp_path)
    run(path)
    once = path.read_text(encoding="utf-8")
    run(path)
    assert path.read_text(encoding="utf-8") == once
    assert "không có gì thay đổi" in capsys.readouterr().out
    fresh = write_splits(tmp_path)
    original = fresh.read_text(encoding="utf-8")
    run(fresh, "--dry-run")
    assert fresh.read_text(encoding="utf-8") == original
    assert "dry-run, chưa ghi" in capsys.readouterr().out


def test_bad_set_and_unknown_action(tmp_path, capsys):
    with pytest.raises(SystemExit):
        US.main(["--action", "x", "--set", "khong_co_dau_bang", "--splits", str(tmp_path)])
    path = write_splits(tmp_path)
    US.main(["--action", "999_none", "--set", "a=b", "--splits", str(path)])
    assert "Không có sequence nào" in capsys.readouterr().out
