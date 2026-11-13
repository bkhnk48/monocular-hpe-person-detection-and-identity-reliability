"""
Hiệu chỉnh hệ số (alpha, gamma, C_FP, ...) nhiều lần, TÍCH LUỸ dữ liệu qua các lần.

Mỗi lần chạy:
  1. Đọc tập validation trong manifest (configs/splits.json).
  2. Đối chiếu với sổ đăng ký configs/tuning/registry.json:
       - sequence mới             -> dựng mạng, lưu đại lượng thô vào cache, đăng ký
       - đã có (cùng khoá)        -> bỏ qua, dùng cache (không tính, không đếm 2 lần)
       - đã có nhưng nhãn đổi     -> tính lại với mapping_frames.csv mới
       - trùng một sequence test  -> từ chối
       - đã bị loại (--exclude)   -> bỏ qua
       - "skip": true trong manifest -> không thêm, không cập nhật; nếu đã đăng ký
                                    từ lần trước thì cache vẫn được dùng (bỏ hẳn: --exclude)
     Khoá của sequence = Drive File ID của wham_pkl ("drive:<id>"),
     hoặc đường dẫn pkl cục bộ ("local:<path>") nếu manifest ghi đường dẫn.
  3. Quét lưới hệ số trên TOÀN BỘ sequence đã đăng ký (mọi lần cộng lại),
     chọn điểm ở giữa vùng ổn định (điểm trung bình với các ô lân cận cao nhất).
  4. Ghi configs/tuning/round_<n>.yaml (dùng được làm --config cho build_network)
     và outputs/tuning/round_<n>_results.csv.

C_FN được cố định (mặc định 1.0): nhân mọi chi phí với cùng một số thì nghiệm không đổi.

Chạy từ thư mục gốc repo:
  python3 -m pose_assoc.tune                         # chạy một lần hiệu chỉnh
  python3 -m pose_assoc.tune --list                  # xem sổ đăng ký và các lần
  python3 -m pose_assoc.tune --exclude mpi_S8_Seq1_cam8
  python3 -m pose_assoc.tune --grid alpha=0.5,1,2 --grid gamma=0,0.5
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter

from .build_network import REPO, is_skipped, load_inputs, load_params, mapping_module
from .costs import CostParams
from .evaluation import load_truth_by_person, require_truth_by_person
from .network import apply_params, build_network
from .solvers.multi_mcf import solve_network

TUNING_DIR = REPO / "configs" / "tuning"
REGISTRY = TUNING_DIR / "registry.json"
CACHE_DIR = REPO / "outputs" / "tuning" / "cache"
RESULTS_DIR = REPO / "outputs" / "tuning"
DEFAULT_GRID = {
    "alpha": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    "c_fp": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    "gamma": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
}
STATUS = ["correct", "correct_miss", "wrong_id", "missed"]

# ============================================================
# 1. SỔ ĐĂNG KÝ
# ============================================================


def seq_key(seq: dict) -> str:
    if seq.get("wham_pkl_is_drive"):
        return f"drive:{seq['wham_pkl']}"
    return f"local:{Path(seq['wham_pkl']).resolve()}"

def gt_signature(seq: dict) -> str:
    """Dấu vân tay nội dung GT, không phụ thuộc máy."""
    gt = seq["gt"]
    if gt.get("path_is_drive"):
        return f"{gt['type']}:drive:{gt['path']}"
    p, h = Path(gt["path"]), hashlib.sha1(gt["type"].encode())
    if p.is_dir():                      # harmony4d: tên + kích thước các tệp
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f"{f.relative_to(p).as_posix()}:{f.stat().st_size}".encode())
    elif p.is_file():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]

def mapping_signature(seq: dict) -> str | None:
    """Dấu vân tay của nhãn gán tay (file nhỏ) để biết nhãn đã bị gán lại chưa."""
    path = seq["out_dir"] / "mapping_frames.csv"
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16] if path.is_file() else None


def load_registry() -> dict:
    if REGISTRY.is_file():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"sequences": {}, "rounds": []}


def save_registry(reg: dict):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".pkl")


def print_registry(reg: dict):
    print(f"\nSổ đăng ký: {REGISTRY}")
    if not reg["sequences"]:
        print("  (trống)")
    for key, e in reg["sequences"].items():
        flag = "  [ĐÃ LOẠI]" if e.get("excluded") else ""
        print(f"  {e['name']:<28} lần {e['added_round']:<3} {key[:48]}{flag}")
    for r in reg["rounds"]:
        p = r["params"]
        print(f"  Lần {r['round']} ({r['date']}): {len(r['sequences'])} sequence, "
              f"alpha={p['alpha']}, gamma={p['gamma']}, C_FP={p['c_fp']}, "
              f"điểm={r['score']:.4f}")


def set_excluded(reg: dict, names: list[str], value: bool):
    for name in names:
        hits = [k for k, e in reg["sequences"].items() if name in (e["name"], k)]
        if not hits:
            print(f"  Không có '{name}' trong sổ đăng ký.")
        for k in hits:
            reg["sequences"][k]["excluded"] = value
            print(f"  {'Loại' if value else 'Dùng lại'}: {reg['sequences'][k]['name']}")

# ============================================================
# 2. CHUẨN BỊ DỮ LIỆU (MỘT LẦN CHO MỖI SEQUENCE)
# ============================================================


def node_status(net, truth: dict) -> np.ndarray:
    """Mã trạng thái của mỗi nút nếu được chọn: -1 nguồn/đích, 0..3 theo STATUS."""
    require_truth_by_person(truth)
    out = np.full(len(net.nodes), -1)
    for n in net.nodes:
        ids = truth.get((n["frame"], n["person"]), set())
        if n["kind"] == "det":
            out[n["id"]] = 0 if n["body_id"] in ids else 2
        elif n["kind"] == "miss":
            out[n["id"]] = 3 if ids else 1
    return out


def prepare_sequence(mod, manifest: dict, seq: dict, base: CostParams) -> dict:
    dets, gt = load_inputs(mod, manifest, seq)
    net = build_network(dets, gt, base)
    truth = load_truth_by_person(seq["out_dir"] / "mapping_frames.csv")
    data = {"name": seq["name"], "net": net, "status": node_status(net, truth)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path(seq_key(seq)), "wb") as f:
        pickle.dump(data, f)
    print(f"    {len(net.nodes)} nút, {len(net.tails)} cung, K = {len(net.persons)}")
    return data


def classify(seq: dict, key: str, reg: dict, test_keys: set) -> str:
    if key in test_keys:
        return "test"
    if is_skipped(seq):
        return "skip"
    e = reg["sequences"].get(key)
    if e is None:
        return "new" if mapping_signature(seq) else "unlabeled"
    if e.get("excluded"):
        return "excluded"
    sig = mapping_signature(seq)
    old_gt = e.get("gt_signature")
    if old_gt is not None and old_gt != gt_signature(seq):
        return "gt_changed"
    # Không có nhãn ở chỗ mới (ví dụ điền lại dưới tên khác) -> vẫn là sequence đã có
    return "changed" if sig is not None and (sig != e["mapping_sig"])  else "known"


MESSAGES = {"new": "mới            -> dựng mạng",
            "changed": "NHÃN ĐÃ ĐỔI    -> dựng lại với nhãn mới",
            "gt_changed": "GT ĐÃ ĐỔI        -> dựng lại với GT mới",
            "known": "ĐÃ CÓ (lần {r}) -> bỏ qua, dùng dữ liệu đã lưu",
            "excluded": "ĐÃ BỊ LOẠI     -> bỏ qua",
            "unlabeled": "chưa có mapping_frames.csv -> bỏ qua (gán nhãn trước)",
            "skip": "skip=true      -> bỏ qua, không thêm hay cập nhật",
            "test": "TRÙNG TẬP TEST -> TỪ CHỐI (không chỉnh trên dữ liệu test)"}
SKIP_REGISTERED = ("{msg}; đã đăng ký từ lần {r} nên dữ liệu đã lưu vẫn được dùng "
                   "(bỏ hẳn: --exclude {name})")


def state_message(state: str, seq: dict, old: dict) -> str:
    msg = MESSAGES[state].format(r=old.get("added_round"))
    if state == "skip" and old and not old.get("excluded"):
        msg = SKIP_REGISTERED.format(msg=msg, r=old.get("added_round"), name=seq["name"])
    return msg


def register_validation(mod, manifest: dict, reg: dict, base: CostParams, rnd: int) -> int:
    """Xử lý tập validation; trả về số sequence mới hoặc đổi nhãn."""
    test_keys = {seq_key(s) for s in manifest.get("test", [])}
    seen, changed = set(), 0
    for seq in manifest.get("validation", []):
        key = seq_key(seq)
        if key in seen:
            print(f"  {seq['name']:<28} lặp lại trong manifest -> bỏ qua")
            continue
        seen.add(key)
        state = classify(seq, key, reg, test_keys)
        old = reg["sequences"].get(key, {})
        print(f"  {seq['name']:<28} " + state_message(state, seq, old))
        if state in ("new", "changed", "gt_changed"):
            prepare_sequence(mod, manifest, seq, base)
            reg["sequences"][key] = {
                "name": seq["name"], "gt": str(seq["gt"]["path"]),
                "mapping_sig": mapping_signature(seq), "gt_signature": gt_signature(seq), "excluded": False,
                "added_round": old.get("added_round", rnd), "updated_round": rnd}
            changed += 1
    return changed


def warn_test_leak(manifest: dict, reg: dict):
    for s in manifest.get("test", []):
        e = reg["sequences"].get(seq_key(s))
        if e and not e.get("excluded"):
            print(f"  CẢNH BÁO: '{s['name']}' (tập test) đã dùng để chỉnh ở lần "
                  f"{e['added_round']} -> kết quả test của nó không còn khách quan.")


def load_cached(reg: dict) -> list[dict]:
    data = []
    for key, e in reg["sequences"].items():
        if not e.get("excluded"):
            with open(cache_path(key), "rb") as f:
                data.append(pickle.load(f))
    return data

# ============================================================
# 3. QUÉT LƯỚI
# ============================================================


def score_one(d: dict, p: CostParams, wrong_penalty: float) -> tuple[float, np.ndarray, bool]:
    net = apply_params(d["net"], p)
    with warnings.catch_warnings():          # đếm ở evaluate_params thay vì in ra
        warnings.simplefilter("ignore")
        flow, info = solve_network(net)
    chosen = d["status"][np.asarray(net.heads)[flow == 1]]
    counts = np.bincount(chosen[chosen >= 0], minlength=4)
    n = counts.sum()
    return (counts[0] + counts[1] - wrong_penalty * counts[2]) / n, counts, info["optimal"]


def evaluate_params(data: list[dict], p: CostParams, wrong_penalty: float) -> dict:
    """Điểm = trung bình các sequence (mỗi sequence trọng số như nhau)."""
    per_seq, total, non_optimal = {}, np.zeros(4, int), 0
    for d in data:
        s, c, opt = score_one(d, p, wrong_penalty)
        per_seq[d["name"]] = s
        total += c
        non_optimal += not opt
    row = {"score": float(np.mean(list(per_seq.values()))),
           "non_optimal": non_optimal}   # số sequence MILP hết giờ, chưa chứng minh tối ưu
    row.update({k: int(v) for k, v in zip(STATUS, total)})
    row.update({f"score[{k}]": v for k, v in per_seq.items()})
    return row


def grid_search(data: list[dict], base: CostParams, grid: dict, wrong_penalty: float):
    names = list(grid)
    combos = list(itertools.product(*grid.values()))
    rows, t0 = [], time.time()
    for i, values in enumerate(combos, 1):
        p = replace(base, **dict(zip(names, values)))
        rows.append({**dict(zip(names, values)), **evaluate_params(data, p, wrong_penalty)})
        if i % 20 == 0 or i == len(combos):
            print(f"\r  Đã thử {i}/{len(combos)} bộ hệ số ({time.time() - t0:.0f} s)",
                  end="", flush=True)
    print()
    return pd.DataFrame(rows)


def pick_stable(df: pd.DataFrame, grid: dict) -> pd.Series:
    """Chọn ô có điểm trung bình 3x3x... lân cận cao nhất (vùng ổn định)."""
    shape = [len(v) for v in grid.values()]
    scores = df["score"].to_numpy().reshape(shape)
    smooth = uniform_filter(scores, size=3, mode="nearest")
    df["smoothed"] = smooth.reshape(-1)
    best = df.sort_values(["smoothed", "score"], ascending=False).iloc[0]
    return best

# ============================================================
# 4. GHI KẾT QUẢ
# ============================================================


def association_block(p: CostParams) -> dict:
    return {"weights": {"pose": p.alpha, "temporal": p.gamma},
            "costs": {"false_positive": p.c_fp, "false_negative": p.c_fn,
                      "disappear": p.c_disappear, "reappear": p.c_reappear},
            "scales": {"pose_mm": p.pose_scale_mm, "motion_m": p.motion_scale_m},
            "boundary": {"start_detected": p.start_detected, "start_missed": p.start_missed,
                         "end_detected": p.end_detected, "end_missed": p.end_missed}}


def write_round(rnd: int, p: CostParams, best: pd.Series, data: list[dict], df, args):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"round": rnd, "date": datetime.now().isoformat(timespec="seconds"),
           "sequences": [d["name"] for d in data], "wrong_penalty": args.wrong_penalty,
           "score": float(best["score"]), "smoothed_score": float(best["smoothed"]),
           "association": association_block(p)}
    path = TUNING_DIR / f"round_{rnd}.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    df.to_csv(RESULTS_DIR / f"round_{rnd}_results.csv", index=False)
    return path


def print_comparison(prev: dict | None, p: CostParams, best: pd.Series,
                     prev_now: dict | None, n_seq: int):
    print(f"\n  {'':<16}{'lần trước':>12}{'lần này':>12}")
    for name in ("alpha", "gamma", "c_fp", "c_disappear", "c_reappear"):
        old = f"{prev['params'][name]:.3g}" if prev else "-"
        print(f"  {name:<16}{old:>12}{getattr(p, name):>12.3g}")
    old = f"{prev_now['score']:.2%}" if prev_now else "-"
    print(f"  {'điểm':<16}{old:>12}{best['score']:>12.2%}   (trên {n_seq} sequence hiện có)")
    print(f"  {'điểm vùng 3x3':<16}{'':>12}{best['smoothed']:>12.2%}")
    for col in best.index:
        if col.startswith("score["):
            print(f"    {col[6:-1]:<30} {best[col]:.2%}")

# ============================================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ============================================================


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_grid(items: list[str] | None) -> dict:
    grid = {k: list(v) for k, v in DEFAULT_GRID.items()}
    for item in items or []:
        name, values = item.split("=", 1)
        name = name.strip().replace("-", "_")
        if name not in asdict(CostParams()):
            raise SystemExit(f"Hệ số không tồn tại: {name}")
        grid[name] = [float(v) for v in values.split(",") if v.strip()]
    return grid


def parse_args():
    p = argparse.ArgumentParser(description="Hiệu chỉnh hệ số tích luỹ trên tập validation")
    p.add_argument("--manifest", type=Path, default=REPO / "configs" / "splits.json")
    p.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml",
                   help="hệ số gốc cho các hệ số không quét")
    p.add_argument("--grid", action="append", help="ví dụ alpha=0.5,1,2 (lặp được)")
    p.add_argument("--wrong-penalty", type=float, default=0.0,
                   help="điểm = (đúng - λ*wrong_id)/N; mặc định 0 = độ chính xác")
    p.add_argument("--exclude", nargs="+", help="loại sequence (tên hoặc khoá) khỏi sổ")
    p.add_argument("--include", nargs="+", help="dùng lại sequence đã loại")
    p.add_argument("--list", action="store_true", help="chỉ in sổ đăng ký")
    p.add_argument("--force", action="store_true", help="chỉnh lại kể cả không có dữ liệu mới")
    for name in ("alpha", "gamma", "c_fp", "c_fn", "c_disappear", "c_reappear"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, dest=name,
                       help=argparse.SUPPRESS)
    return p.parse_args()


def run_round(mod, manifest: dict, reg: dict, args) -> None:
    rnd = len(reg["rounds"]) + 1
    base = load_params(args.config, args)
    print(f"\nLần hiệu chỉnh {rnd}. Tập validation trong manifest:")
    changed = register_validation(mod, manifest, reg, base, rnd)
    warn_test_leak(manifest, reg)
    save_registry(reg)
    if changed == 0 and reg["rounds"] and not args.force:
        print("\n  Không có dữ liệu mới so với lần trước -> không chỉnh lại "
              "(dùng --force để chỉnh lại).")
        return
    data = load_cached(reg)
    if not data:
        raise SystemExit("Chưa có sequence nào đủ điều kiện để hiệu chỉnh.")
    grid = parse_grid(args.grid)
    n_combo = int(np.prod([len(v) for v in grid.values()]))
    print(f"\n  Chỉnh trên {len(data)} sequence ({', '.join(d['name'] for d in data)}), "
          f"{n_combo} bộ hệ số, C_FN = {base.c_fn} cố định")
    df = grid_search(data, base, grid, args.wrong_penalty)
    best = pick_stable(df, grid)
    p = replace(base, **{k: float(best[k]) for k in grid})
    prev = reg["rounds"][-1] if reg["rounds"] else None
    prev_now = (evaluate_params(data, CostParams(**prev["params"]), args.wrong_penalty)
                if prev else None)
    print_comparison(prev, p, best, prev_now, len(data))
    path = write_round(rnd, p, best, data, df, args)
    reg["rounds"].append({"round": rnd, "date": datetime.now().isoformat(timespec="seconds"),
                          "sequences": [d["name"] for d in data], "params": asdict(p),
                          "score": float(best["score"]), "file": rel(path)})
    save_registry(reg)
    print(f"\n  Đã lưu: {path}")
    print(f"  Đã lưu: {RESULTS_DIR / f'round_{rnd}_results.csv'}")
    print(f"  Dùng: python3 -m pose_assoc.build_network --seq <tên> --config {rel(path)}")


def main():
    args = parse_args()
    reg = load_registry()
    if args.exclude or args.include:
        set_excluded(reg, args.exclude or [], True)
        set_excluded(reg, args.include or [], False)
        save_registry(reg)
        args.force = True        # tập dữ liệu đã đổi -> phải chỉnh lại
    if args.list:
        print_registry(reg)
        return
    mod = mapping_module()
    run_round(mod, mod.load_manifest(args.manifest), reg, args)


if __name__ == "__main__":
    main()
