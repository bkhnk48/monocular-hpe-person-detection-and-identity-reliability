#!/usr/bin/env python3
"""
Đặt giá trị cho các trường của mọi sequence thuộc một action X trong configs/splits.json.

Sequence thuộc action X khi "name" có dạng h4d_<X>_cam<số>. Mỗi --set có dạng
khoá=giá trị; khoá lồng nhau viết bằng dấu chấm (gt.path_2d). Trường chưa có thì được
thêm vào cuối đối tượng chứa nó; giá trị luôn là chuỗi (File ID của Drive).

Chỉ đoạn văn bản của các phần tử khớp được viết lại (json, thụt 2 dấu cách); phần còn
lại của tệp, kể cả chỗ định dạng tay, giữ nguyên từng ký tự.

Chạy:
  python3 scripts/update_splits.py --action 009_ballroom2 \\
      --set gt.path_2d=17ZH-TzSEqnIO6CxncNqrd1Rt6zsc0dUd \\
      --set colmap_folder_ID=1nf4LpjO_aIr_GtHmaizim0berMEGa6uq --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("validation", "test")


def parse_set(item: str) -> tuple[list[str], str]:
    """'gt.path_2d=ID' -> (['gt', 'path_2d'], 'ID')."""
    key, sep, value = item.partition("=")
    if not sep or not key or any(not k for k in key.split(".")):
        raise argparse.ArgumentTypeError(f"--set cần dạng khoá=giá trị, nhận '{item}'")
    return key.split("."), value


def apply_sets(entry: dict, sets) -> list[str]:
    """Đặt các giá trị vào entry; trả về mô tả các trường đã đổi."""
    changes = []
    for keys, value in sets:
        target = entry
        for k in keys[:-1]:
            target = target.setdefault(k, {})
            if not isinstance(target, dict):
                raise ValueError(f"{entry['name']}: '{k}' không phải đối tượng JSON")
        old = target.get(keys[-1], "<chưa có>")
        if old != value:
            target[keys[-1]] = value
            changes.append(f"{'.'.join(keys)}: {old} -> {value}")
    return changes


def entry_spans(text: str, names: set) -> dict:
    """{name: (vị trí '{', vị trí sau '}')} của các phần tử có name thuộc names."""
    decoder, spans = json.JSONDecoder(), {}
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        if m.group(1) in names:
            start = text.rfind("{", 0, m.start())
            obj, end = decoder.raw_decode(text, start)
            if obj.get("name") != m.group(1):
                raise ValueError(f"Không định vị được phần tử '{m.group(1)}'")
            spans[m.group(1)] = (start, end)
    return spans


def update_text(text: str, action: str, sets) -> tuple[str, list[str]]:
    """Văn bản mới của splits.json và nhật ký thay đổi."""
    data = json.loads(text)
    pattern = re.compile(rf"h4d_{re.escape(action)}_cam\d+")
    matched = {e["name"]: e for split in SPLITS for e in data.get(split, [])
               if pattern.fullmatch(e["name"])}
    log, new_entries = [], {}
    for name, entry in matched.items():
        changes = apply_sets(entry, sets)
        log.append(f"{name}: " + ("; ".join(changes) if changes else "không đổi"))
        if changes:
            new_entries[name] = entry
    spans = entry_spans(text, set(new_entries))
    for name in sorted(spans, key=lambda n: spans[n][0], reverse=True):
        start, end = spans[name]
        line_start = text.rfind("\n", 0, start) + 1
        indent = text[line_start:start]
        dumped = json.dumps(new_entries[name], ensure_ascii=False, indent=2)
        text = text[:start] + dumped.replace("\n", "\n" + indent) + text[end:]
    return text, log if matched else [f"Không có sequence nào dạng h4d_{action}_camY"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--action", required=True, help="X trong h4d_<X>_camY, ví dụ 009_ballroom2")
    ap.add_argument("--set", dest="sets", type=parse_set, action="append", required=True,
                    metavar="KHOÁ=GIÁ_TRỊ", help="ví dụ gt.path_2d=<ID>; lặp lại được")
    ap.add_argument("--splits", type=Path, default=ROOT / "configs" / "splits.json")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in thay đổi, không ghi tệp")
    args = ap.parse_args(argv)

    text = args.splits.read_text(encoding="utf-8")
    new_text, log = update_text(text, args.action, args.sets)
    print("\n".join(log))
    json.loads(new_text)                     # chắc chắn vẫn là JSON hợp lệ trước khi ghi
    changed = new_text != text
    if changed and not args.dry_run:
        args.splits.write_text(new_text, encoding="utf-8")
    state = "không có gì thay đổi" if not changed else (
        "dry-run, chưa ghi" if args.dry_run else f"đã ghi {args.splits}")
    print(f"=> {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
