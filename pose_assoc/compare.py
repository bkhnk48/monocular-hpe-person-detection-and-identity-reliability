
#pose_assoc/compare.py
#CLI dưới đây đọc danh sách sequence từ configs/splits.json, bỏ qua sequence có skip: true, 
# chỉ chạy sequence có mapping_frames.csv, rồi xuất một dòng cho mỗi (method, sequence) 
# và một dòng macro-average cho mỗi method.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from .baselines import (
    greedy_per_frame,
    hungarian_per_frame,
    independent_dp,
    original_wham_id_majority,
    assignment_from_arc_flow,
)
from .evaluation import (
    evaluate_assignment,
    load_truth,
    macro_average,
)
from .network import apply_params, attach_positions, build_network
from .solvers.multi_mcf import solve_network


METHOD_ORDER = ["B6", "B1", "B2", "B3", "B4"]


def _load_manifest(path: Path):
    # Dùng chung loader với build_network/wham_gt_mapping: load_inputs() cần các trường
    # do nó tạo (seq["out_dir"], gt.path đã resolve, wham_pkl_is_drive, smpl_model).
    from .build_network import mapping_module

    manifest = mapping_module().load_manifest(path.resolve())

    # splits.json quy định base_dir tương đối với chính file manifest.
    configured_base = Path(manifest.get("base_dir", "."))
    if not configured_base.is_absolute():
        configured_base = (path.resolve().parent / configured_base).resolve()

    manifest["_base_dir_resolved"] = configured_base
    manifest["_output_dir_resolved"] = Path(manifest["output_dir"])

    return manifest


def _load_params(config_path: Path):
    from .costs import CostParams

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    association = config.get("association", config)
    return CostParams.from_config(association)


def _resolve_path(value, base_dir: Path):
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _sequence_name(seq):
    return str(seq.get("name") or seq.get("id") or seq.get("sequence"))


def _get_split_sequences(manifest, split):
    sequences = manifest.get(split, [])
    if not isinstance(sequences, list):
        raise ValueError(f"Manifest mục '{split}' phải là một list.")
    return sequences


def _mapping_file(manifest, seq):
    # Ưu tiên out_dir do wham_gt_mapping.load_manifest tạo ra.
    if seq.get("out_dir"):
        out_dir = Path(seq["out_dir"])
        if not out_dir.is_absolute():
            out_dir = (
                manifest["_base_dir_resolved"] / out_dir
            ).resolve()
    else:
        out_dir = (
            manifest["_output_dir_resolved"] / _sequence_name(seq)
        )

    return out_dir / "mapping_frames.csv"


def _load_sequence_inputs(seq, manifest, params):
    """
    Dùng loader hiện có trong build_network.py để tránh tạo một
    pipeline đọc WHAM/GT thứ hai không nhất quán.
    """
    from .build_network import load_inputs, load_positions, mapping_module

    mod = mapping_module()

    # load_inputs hiện tại nhận manifest và seq, tự gọi loader WHAM/GT.
    detections, gt = load_inputs(mod, manifest, seq)

    net = build_network(detections, gt, params)
    # Cả 5 phương pháp dùng chung mạng này, nên cùng thấy C_V. B1/B2 chỉ đọc chi phí
    # nút nên C_V (chi phí cung) không đổi kết quả của chúng; B3/B4 thì có.
    if params.eta:
        apply_params(attach_positions(net, load_positions(mod, manifest, seq)), params)
    return net


def _run_one_sequence(seq, manifest, params, mapping_path, time_limit):
    name = _sequence_name(seq)

    net = _load_sequence_inputs(seq, manifest, params)
    truth = load_truth(mapping_csv=str(mapping_path))

    outputs = []

    # B6: original WHAM ID + majority vote.
    b6_assignment, body_to_person = original_wham_id_majority(net)
    outputs.append(
        evaluate_assignment(
            net,
            b6_assignment,
            truth,
            method="B6",
            sequence=name,
        )
    )

    # B1: greedy per frame.
    outputs.append(
        evaluate_assignment(
            net,
            greedy_per_frame(net),
            truth,
            method="B1",
            sequence=name,
        )
    )

    # B2: Hungarian per frame.
    outputs.append(
        evaluate_assignment(
            net,
            hungarian_per_frame(net),
            truth,
            method="B2",
            sequence=name,
        )
    )

    # B3: independent shortest paths, không có shared-detection constraint.
    outputs.append(
        evaluate_assignment(
            net,
            independent_dp(net),
            truth,
            method="B3",
            sequence=name,
        )
    )

    # B4: MC-MCF hiện tại.
    flow, solver_info = solve_network(net, time_limit=time_limit)
    b4_assignment = assignment_from_arc_flow(net, flow)

    outputs.append(
        evaluate_assignment(
            net,
            b4_assignment,
            truth,
            method="B4",
            sequence=name,
            solver_info=solver_info,
        )
    )

    return outputs


def _print_comparison(table: pd.DataFrame):
    metrics = [
        "accuracy",
        "idf1",
        "det_f1",
        "mean_pa_mpjpe_mm",
        "id_changes",
    ]

    macro = table[table["sequence"] == "MACRO_AVERAGE"].copy()
    if macro.empty:
        print("\nKhông có macro-average để in.")
        return

    macro = macro.set_index("method")

    print("\n=== MACRO AVERAGE ===")
    print(
        macro.reindex(METHOD_ORDER)[metrics].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    baseline_methods = ["B6", "B1", "B2", "B3"]
    baseline_scores = macro.reindex(baseline_methods)

    # Chọn baseline tốt nhất theo IDF1; nếu bằng nhau thì xét accuracy.
    best_baseline = max(
        baseline_methods,
        key=lambda method: (
            float(baseline_scores.loc[method, "idf1"]),
            float(baseline_scores.loc[method, "accuracy"]),
        ),
    )

    if "B4" not in macro.index or best_baseline not in macro.index:
        return

    print(f"\n=== B4 so với baseline tốt nhất: {best_baseline} ===")
    for metric in metrics:
        b4 = macro.loc["B4", metric]
        baseline = macro.loc[best_baseline, metric]

        if pd.isna(b4) or pd.isna(baseline):
            delta = np.nan
        else:
            delta = float(b4 - baseline)

        print(f"{metric:24s}: {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="So sánh B6/B1/B2/B3/B4 trên cùng mapping thủ công."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )
    parser.add_argument(
        "--manifest",
        default="configs/splits.json",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Ví dụ: configs/tuning/round_2.yaml",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()

    manifest = _load_manifest(manifest_path)
    params = _load_params(config_path)

    sequences = _get_split_sequences(manifest, args.split)

    all_results = []

    for seq in sequences:
        name = _sequence_name(seq)

        if seq.get("skip", False):
            print(f"[SKIP] {name}: skip=true")
            continue

        mapping_path = _mapping_file(manifest, seq)

        if not mapping_path.is_file():
            print(f"[SKIP] {name}: thiếu {mapping_path}")
            continue

        print(f"\n[RUN] {name}")

        try:
            results = _run_one_sequence(
                seq=seq,
                manifest=manifest,
                params=params,
                mapping_path=mapping_path,
                time_limit=args.time_limit,
            )
            all_results.extend(results)

            for result in results:
                print(
                    f"  {result['method']}: "
                    f"accuracy={result['accuracy']:.4f}, "
                    f"IDF1={result['idf1']:.4f}, "
                    f"det-F1={result['det_f1']:.4f}"
                )

        except Exception as exc:
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            raise

    if not all_results:
        raise RuntimeError(
            f"Không có sequence hợp lệ trong split '{args.split}'."
        )

    results_df = pd.DataFrame(all_results)
    results_df["_order"] = results_df["method"].map(
        {method: i for i, method in enumerate(METHOD_ORDER)}
    )
    results_df = results_df.sort_values(
        ["_order", "sequence"]
    ).drop(columns="_order")

    macro_df = macro_average(results_df)
    table = pd.concat([results_df, macro_df], ignore_index=True)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/compare") / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "table.csv"
    table.to_csv(output_csv, index=False)

    print(f"\nĐã ghi: {output_csv}")
    _print_comparison(table)


if __name__ == "__main__":
    main()