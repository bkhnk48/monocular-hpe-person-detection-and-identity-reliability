"""
Dựng mạng time-expanded, giải min-cost flow (1 hay nhiều luồng) và chấm điểm.

Đầu vào lấy từ manifest (configs/splits.json) theo tên sequence:
  - RA-WHAM.csv  : outputs/mapping/<seq>/RA-WHAM.csv (tự tạo từ pkl nếu chưa có)
  - Ground Truth : annot.mat (MPI, theo camera/person trong manifest) hoặc poses3d/*.npy
  - hệ số        : configs/default.yaml, mục association (ghi đè bằng --alpha, --gamma, ...)
  - đáp án       : outputs/mapping/<seq>/mapping_frames.csv (nếu có thì chấm điểm)

Đầu ra (outputs/network/<seq>/):
  nodes.csv, arcs.csv   toàn bộ mạng (arcs.csv có cột flow = nghiệm)
  assignment.csv        mỗi (frame, người GT): body_id được chọn hoặc trống (missed)
  summary.json          kích thước mạng, tham số, hàm mục tiêu, điểm số

Chạy từ thư mục gốc repo:
  python3 -m pose_assoc.build_network --seq mpi_S8_Seq1_cam0
  python3 -m pose_assoc.build_network --split validation
  python3 -m pose_assoc.build_network --seq h4d_041_grappling2_cam07 --alpha 1.5 --gamma 0.3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .costs import LIMBS, CostParams
from .evaluation import assignment_table, evaluate, load_truth, objective_breakdown
from .network import arcs_frame, build_network, nodes_frame
from .solvers.multi_mcf import solve_network

REPO = Path(__file__).resolve().parent.parent
GT_LIMB_SLICE = {"mpi13": slice(1, 13), "coco17": slice(5, 17)}
OVERRIDES = {"alpha": "alpha", "gamma": "gamma", "c_fp": "c_fp", "c_fn": "c_fn",
             "c_disappear": "c_disappear", "c_reappear": "c_reappear"}


def mapping_module():
    """Dùng lại bộ đọc manifest / GT / pkl của scripts/wham_gt_mapping.py."""
    path = REPO / "scripts" / "wham_gt_mapping.py"
    spec = importlib.util.spec_from_file_location("wham_gt_mapping", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_params(config: Path, args) -> CostParams:
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = CostParams.from_config(cfg.get("association", {}))
    for arg, attr in OVERRIDES.items():
        value = getattr(args, arg)
        if value is not None:
            setattr(params, attr, value)
    return params


def detections_from_csv(df: pd.DataFrame) -> dict:
    """{frame: [(body_id, pose (12,3))]} — khớp tương đối pelvis là đủ cho PA-MPJPE."""
    cols = [f"{j}_{a}" for j in LIMBS for a in "xyz"]
    poses = df[cols].to_numpy(float).reshape(len(df), len(LIMBS), 3)
    dets: dict = {}
    for f, b, pose in zip(df["frame_id"].astype(int), df["body_id"].astype(int), poses):
        dets.setdefault(int(f), []).append((int(b), pose))
    return dets


def gt_limbs(gt: dict, skel: str) -> dict:
    sl = GT_LIMB_SLICE[skel]
    return {f: {k: pose[sl] for k, pose in people.items()} for f, people in gt.items()}


def find_sequence(manifest: dict, name: str) -> dict:
    for split in ("validation", "test"):
        for seq in manifest.get(split, []):
            if seq["name"] == name:
                return seq
    raise SystemExit(f"Không có sequence '{name}' trong manifest.")


def load_inputs(mod, manifest: dict, seq: dict):
    seq["out_dir"].mkdir(parents=True, exist_ok=True)
    wham = mod.load_wham_csv(seq, manifest["smpl_model"])
    print("  Đọc Ground Truth ...")
    gt, skel = mod.load_gt(seq["gt"])
    return detections_from_csv(wham), gt_limbs(gt, skel)


def export(out: Path, net, flow, assign: pd.DataFrame, summary: dict, full: bool):
    out.mkdir(parents=True, exist_ok=True)
    if full:
        nodes_frame(net).to_csv(out / "nodes.csv", index=False)
        arcs_frame(net, flow).to_csv(out / "arcs.csv", index=False)
    assign.to_csv(out / "assignment.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                 default=float), encoding="utf-8")


def print_summary(s: dict):
    n = s["network"]
    print(f"  Mạng: K = {n['K']} luồng ({', '.join(n['persons'])}), "
          f"{n['layers']} lớp, {n['nodes']} nút, {n['arcs']} cung")
    print(f"  Bộ giải: {s['solver']['method']}, {s['solver']['seconds']} s, "
          f"hàm mục tiêu = {s['solver']['objective']:.4f}")
    ms = s["solver"].get("milp")
    if ms:
        tag = "tối ưu đã chứng minh" if s["solver"]["optimal"] else "CHƯA chứng minh tối ưu"
        print(f"    HiGHS: {ms['status_name']} ({tag}), mip_gap {ms['mip_gap']}, "
              f"{ms['mip_node_count']} nút nhánh cận")
    o = s["objective"]
    print(f"    pose {o['pose']:.3f} | FP {o['false_positive']:.3f} | FN {o['false_negative']:.3f}"
          f" | thời gian {o['temporal_and_gaps']:.3f} | biên {o['boundary']:.3f}")
    m = s.get("metrics")
    if m:
        print(f"  So với mapping gán tay: đúng {m['accuracy']:.2%} "
              f"(correct {m['correct']}, correct_miss {m['correct_miss']}, "
              f"wrong_id {m['wrong_id']}, missed {m['missed']}), đổi id {m['id_changes']} lần")
    else:
        print("  Chưa có mapping_frames.csv -> bỏ qua chấm điểm.")


def run_sequence(mod, manifest: dict, seq: dict, args):
    print(f"\n=== {seq['name']} ===")
    dets, gt = load_inputs(mod, manifest, seq)
    params = load_params(args.config, args)
    net = build_network(dets, gt, params)
    flow, info = solve_network(net, args.time_limit)
    assign = assignment_table(net, flow)
    summary = {"sequence": seq["name"], "params": asdict(params), "solver": info,
               "network": {"K": len(net.persons), "persons": net.persons,
                           "layers": int(sum(n["kind"] == "miss" for n in net.nodes)),
                           "nodes": len(net.nodes), "arcs": len(net.tails)},
               "objective": objective_breakdown(net, flow, params)}
    truth_path = seq["out_dir"] / "mapping_frames.csv"
    if truth_path.is_file():
        assign, summary["metrics"] = evaluate(assign, load_truth(truth_path))
    out = args.out or REPO / "outputs" / "network" / seq["name"]
    export(Path(out), net, flow, assign, summary, not args.no_network_csv)
    print_summary(summary)
    print(f"  Kết quả: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Dựng mạng time-expanded và giải min-cost flow")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq", help="tên sequence trong manifest")
    g.add_argument("--split", choices=["validation", "test"], help="chạy cả tập")
    p.add_argument("--manifest", type=Path, default=REPO / "configs" / "splits.json")
    p.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    p.add_argument("--out", type=Path, help="thư mục kết quả (mặc định outputs/network/<seq>)")
    for name in OVERRIDES:
        p.add_argument(f"--{name.replace('_', '-')}", type=float, dest=name,
                       help="ghi đè hệ số trong config")
    p.add_argument("--time-limit", type=float, default=300.0, help="giới hạn MILP (giây)")
    p.add_argument("--no-network-csv", action="store_true", help="không xuất nodes/arcs.csv")
    return p.parse_args()


def main():
    args = parse_args()
    mod = mapping_module()
    manifest = mod.load_manifest(args.manifest)
    seqs = ([find_sequence(manifest, args.seq)] if args.seq
            else manifest.get(args.split, []))
    if args.out and len(seqs) > 1:
        raise SystemExit("--out chỉ dùng với một --seq.")
    for seq in seqs:
        if mod.sequence_status(seq).startswith("THIẾU"):
            print(f"\n=== {seq['name']} === thiếu dữ liệu -> bỏ qua")
            continue
        run_sequence(mod, manifest, seq, args)


if __name__ == "__main__":
    main()
