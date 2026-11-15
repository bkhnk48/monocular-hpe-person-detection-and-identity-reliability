"""scripts/help phải chạy được và chỉ nhắc tới lệnh, tuỳ chọn có thật.

Chạy:  python3 -m pytest -q tests/test_help.py

Mỗi dòng lệnh `python3 scripts/<x>.py ...` hoặc `python3 -m pose_assoc.<y> ...` trong
hướng dẫn được đối chiếu với `--help` của chính chương trình đó. Nhờ vậy, khi đổi tên
hay bỏ một tuỳ chọn, test này báo hướng dẫn đã cũ.
"""

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELP = ROOT / "scripts" / "help"
TITLES = {
    "mapping": "1. TẠO MAPPING",
    "overlay": "2. TẠO VIDEO OVERLAY",
    "tune": "3. HIỆU CHỈNH THAM SỐ",
    "test": "4. GIẢI BÀI TOÁN TỐI ƯU",
}
COMMAND = re.compile(r"^\s*python3 (-m pose_assoc\.\w+|scripts/\w+\.py)\b(.*)$")
FLAG = re.compile(r"(?<![\w-])(--[a-z][\w-]*)")


def run_help(*args):
    return subprocess.run([sys.executable, str(HELP), *args], cwd=ROOT,
                          capture_output=True, text=True, check=False)


@lru_cache(maxsize=None)
def tool_help(tool: str) -> str:
    r = subprocess.run([sys.executable, *tool.split(), "--help"], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, (tool, r.stderr[-500:])
    return r.stdout


def test_default_prints_every_section():
    r = run_help()
    assert r.returncode == 0, r.stderr
    for title in TITLES.values():
        assert title in r.stdout


@pytest.mark.parametrize("topic,alias", [("mapping", "map"), ("overlay", "video"),
                                         ("tune", "tuning"), ("test", "solve")])
def test_each_topic_and_alias(topic, alias):
    for name in (topic, alias):
        r = run_help(name)
        assert r.returncode == 0, r.stderr
        assert TITLES[topic] in r.stdout
        others = [t for k, t in TITLES.items() if k != topic]
        assert not any(t in r.stdout for t in others)


def test_unknown_topic_fails():
    r = run_help("khong-co")
    assert r.returncode == 2
    assert "Không có mục" in r.stderr


def test_commands_use_existing_tools_and_flags():
    out = run_help().stdout
    commands = [COMMAND.match(line) for line in out.splitlines()]
    commands = [m for m in commands if m]
    assert len(commands) >= 15
    for m in commands:
        tool, rest = m.group(1), m.group(2)
        known = tool_help(tool)
        for flag in FLAG.findall(rest):
            assert flag in known, f"{tool}: {flag} không có trong --help"


def test_every_flag_mentioned_belongs_to_some_tool():
    out = run_help().stdout
    tools = {m.group(1) for m in map(COMMAND.match, out.splitlines()) if m}
    known = "\n".join(tool_help(t) for t in tools)
    unknown = sorted({f for f in FLAG.findall(out) if f not in known})
    assert not unknown, unknown
