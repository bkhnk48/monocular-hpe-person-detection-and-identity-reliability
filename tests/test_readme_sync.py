"""README.md phải khớp với mã nguồn.

Chạy:  python3 -m pytest -q tests/test_readme_sync.py

Kiểm tra:
  1. Bảng mục 6 của README: mỗi khóa config -> giá trị mặc định trùng với
     CostParams.from_config({}) và với configs/default.yaml.
  2. Mọi trường của CostParams đều có trong bảng (không thiếu tham số).
  3. Mọi đường dẫn `pose_assoc/...`, `scripts/...`, `configs/...` mà README nhắc
     tới đều tồn tại.
"""

import re
from dataclasses import fields
from pathlib import Path

import yaml

from pose_assoc.costs import CostParams

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# khóa config (mục association) -> trường CostParams
KEY_TO_FIELD = {
    "weights.pose": "alpha", "weights.temporal": "gamma",
    "costs.false_positive": "c_fp", "costs.false_negative": "c_fn",
    "costs.disappear": "c_disappear", "costs.reappear": "c_reappear",
    "scales.pose_mm": "pose_scale_mm", "scales.motion_m": "motion_scale_m",
    "temporal.gap_power": "gap_power",
    "boundary.start_detected": "start_detected", "boundary.start_missed": "start_missed",
    "boundary.end_detected": "end_detected", "boundary.end_missed": "end_missed",
}


def readme_table() -> dict:
    """{khóa config: giá trị} từ các dòng bảng có cột khóa trong backtick."""
    out = {}
    for line in README.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.fullmatch(r"`[\w.]+`", cells[-1]):
            num = re.match(r"[-\d.]+", cells[1])
            if num:
                out[cells[-1].strip("`")] = float(num.group())
    return out


def test_table_covers_every_param():
    assert {f.name for f in fields(CostParams)} == set(KEY_TO_FIELD.values())
    assert set(readme_table()) == set(KEY_TO_FIELD)


def test_table_matches_code_defaults():
    p = CostParams.from_config({})
    for key, value in readme_table().items():
        assert getattr(p, KEY_TO_FIELD[key]) == value, key


def test_table_matches_default_yaml():
    assoc = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["association"]
    for key, value in readme_table().items():
        group, name = key.split(".")
        assert assoc[group][name] == value, key


def test_paths_exist():
    paths = set(re.findall(r"`((?:pose_assoc|scripts|configs|tests)/[\w./]+\.\w+)`", README))
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, missing
