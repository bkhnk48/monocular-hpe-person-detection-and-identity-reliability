
#pose_assoc/compare.py
#CLI dưới đây đọc danh sách sequence từ configs/splits.json, bỏ qua sequence có skip: true, 
# chỉ chạy sequence có mapping_frames.csv, rồi xuất một dòng cho mỗi (method, sequence) 
# và một dòng macro-average cho mỗi method.
# Sequence có GT 2D thì chấm thêm theo mức che (MaxIoU khung 2D GT giữa hai người):
# strata.csv, mỗi dòng một (method, sequence, tầng).
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from .baselines import (
    LINK_THRESHOLD,
    LINK_WINDOW,
    VOTE_WINDOW,
    greedy_per_frame,
    hungarian_majority_vote,
    hungarian_per_frame,
    hungarian_vote_2d_tracks,
    independent_dp,
    original_wham_id_majority,
    assignment_from_arc_flow,
)
from .evaluation import (
    HEAVY_IOU,
    STRATA,
    evaluate_assignment,
    load_truth,
    macro_average,
    occlusion_strata,
    stratified_rows,
)
from .network import build_network
from .solvers.multi_mcf import solve_network


# B5 (B2 + majority vote dọc body_id) và B5' (dọc chuỗi nối bằng độ dịch 2D, không dùng
# body_id) đứng ngay sau B2 để dễ so các cột.
METHOD_ORDER = ["B6", "B1", "B2", "B5", "B5'", "B3", "B4"]


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
    from .build_network import attach_image_evidence, load_inputs, mapping_module

    mod = mapping_module()

    # load_inputs hiện tại nhận manifest và seq, tự gọi loader WHAM/GT.
    detections, gt = load_inputs(mod, manifest, seq)

    net = build_network(detections, gt, params)
    # Cả 7 phương pháp dùng chung mạng này. C_V là chi phí cung nên chỉ đổi B3/B4;
    # C_2D là chi phí nút nên đổi cả B1/B2/B5/B5'/B3/B4. B6 (ID gốc của WHAM) chỉ đọc
    # PA-MPJPE thô, không đọc chi phí. B5' đọc thêm toạ độ 2D của WHAM để nối frame.
    return attach_image_evidence(net, mod, manifest, seq, params)


def _occlusion_strata(seq, net):
    """Mức che của mọi (frame, người GT), từ GT 2D; None nếu sequence chưa có GT 2D."""
    from .build_network import load_gt_2d, mapping_module

    mod = mapping_module()
    # Ở đây GT 2D chỉ để chia tầng, nên đọc lỗi thì bỏ phần này (có báo) chứ không dừng
    # cả lượt chạy. Khi beta > 0, attach_image_evidence() đã đọc GT 2D trước và dừng nếu lỗi.
    try:
        if mod.resolve_gt_2d(seq) is None:
            print("  Chưa có GT 2D (gt.path_2d) -> không chấm theo mức che")
            return None
        gt_uv = load_gt_2d(mod, seq)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"  ! Không đọc được GT 2D -> không chấm theo mức che: {exc}")
        return None
    strata = occlusion_strata(net, gt_uv)
    counts = pd.Series([s for _, s in strata.values()]).value_counts()
    print("  Mức che (ô frame × người): "
          + ", ".join(f"{k} {int(counts.get(k, 0))}" for k in STRATA if counts.get(k, 0)))
    return strata


def _run_one_sequence(seq, manifest, params, mapping_path, time_limit, strata_rows=None,
                      vote_window=VOTE_WINDOW, link_window=LINK_WINDOW,
                      link_threshold=LINK_THRESHOLD):
    """Chấm B6..B4 trên một sequence. Truyền list vào strata_rows thì chấm thêm theo
    mức che (khi có GT 2D) và nối các dòng vào đó. vote_window là ω của B5;
    link_window, link_threshold là ω', τ của B5'."""
    from .build_network import load_positions, mapping_module

    name = _sequence_name(seq)

    net = _load_sequence_inputs(seq, manifest, params)
    truth = load_truth(mapping_csv=str(mapping_path))
    strata = _occlusion_strata(seq, net) if strata_rows is not None else None
    det_uv = load_positions(mapping_module(), manifest, seq)

    # B6: ID gốc của WHAM + đa số phiếu; B1: tham lam theo frame; B2: Hungarian theo
    # frame; B5: B2 + majority vote nhãn dọc body_id trong ±ω frame; B5': như B5 nhưng
    # dọc chuỗi nối bằng độ dịch 2D (≤ τ), cửa sổ ±ω'; B3: đường đi ngắn nhất độc lập
    # (không ràng buộc dùng chung detection).
    assignments = {
        "B6": (original_wham_id_majority(net)[0], None),
        "B1": (greedy_per_frame(net), None),
        "B2": (hungarian_per_frame(net), None),
        "B5": (hungarian_majority_vote(net, window=vote_window), None),
        "B5'": (hungarian_vote_2d_tracks(net, det_uv, window=link_window,
                                         threshold=link_threshold,
                                         gap_power=params.gap_power), None),
        "B3": (independent_dp(net), None),
    }
    # B4: MC-MCF.
    flow, solver_info = solve_network(net, time_limit=time_limit)
    assignments["B4"] = (assignment_from_arc_flow(net, flow), solver_info)

    outputs = []
    for method in METHOD_ORDER:
        assignment, info = assignments[method]
        outputs.append(evaluate_assignment(net, assignment, truth, method=method,
                                           sequence=name, solver_info=info))
        if strata is not None:
            strata_rows.extend(stratified_rows(net, assignment, truth, strata,
                                               method=method, sequence=name))
    return outputs


def _print_strata(strata_df: pd.DataFrame):
    """Cộng dồn số đếm của mọi sequence có GT 2D theo (tầng, phương pháp)."""
    pooled = strata_df.groupby(["stratum", "method"], sort=False)[
        ["n_units", "correct", "wrong_id", "missed", "id_changes"]].sum()
    pooled["accuracy"] = pooled["correct"] / (
        pooled["correct"] + pooled["wrong_id"] + pooled["missed"]).replace(0, np.nan)
    n_seq = strata_df.groupby("stratum")["sequence"].nunique()
    print(f"\n=== THEO MỨC CHE (MaxIoU khung 2D GT; tách = 0, chồng ≤ {HEAVY_IOU}, "
          f"chồng nặng > {HEAVY_IOU}) ===")
    print("Cộng dồn mọi sequence có GT 2D; accuracy = correct / (correct + wrong_id + missed).")
    rows = []
    for stratum in STRATA:
        if stratum not in n_seq.index:
            continue
        acc = pooled.loc[stratum]["accuracy"].reindex(METHOD_ORDER)
        chg = pooled.loc[stratum]["id_changes"].reindex(METHOD_ORDER)
        n_units = int(pooled.loc[stratum]["n_units"].iloc[0])
        row = {"tầng": stratum, "sequence": int(n_seq[stratum]), "ô": n_units}
        row.update({m: acc[m] for m in METHOD_ORDER})
        row["B4-B2"] = acc["B4"] - acc["B2"]
        row["B4-B5"] = acc["B4"] - acc["B5"]
        row["B4-B5'"] = acc["B4"] - acc["B5'"]
        # bảng cũ (trước khi có B5, B5') thiếu cột đó -> in "-"
        row["đổi ID B2/B5/B5'/B4"] = "/".join(
            "-" if pd.isna(chg[m]) else str(int(chg[m])) for m in ("B2", "B5", "B5'", "B4"))
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


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

    baseline_methods = ["B6", "B1", "B2", "B5", "B5'", "B3"]
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
        description="So sánh B6/B1/B2/B5/B5'/B3/B4 trên cùng mapping thủ công."
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
    parser.add_argument(
        "--vote-window",
        type=int,
        default=VOTE_WINDOW,
        help=f"ω của B5: bán kính cửa sổ majority vote, tính bằng frame "
             f"(mặc định {VOTE_WINDOW})",
    )
    parser.add_argument(
        "--link-window",
        type=int,
        default=LINK_WINDOW,
        help=f"ω' của B5': bán kính cửa sổ vote dọc chuỗi nối bằng độ dịch 2D, tính bằng "
             f"frame (mặc định {LINK_WINDOW})",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=LINK_THRESHOLD,
        help=f"τ của B5': ngưỡng độ dịch 2D để nối hai detection liên tiếp, cùng đơn vị "
             f"với C_V; 'inf' = không ngưỡng (mặc định {LINK_THRESHOLD})",
    )
    args = parser.parse_args()
    if args.vote_window < 0 or args.link_window < 0:
        parser.error("--vote-window và --link-window phải >= 0")
    if not args.link_threshold >= 0:
        parser.error("--link-threshold phải >= 0")

    manifest_path = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()

    manifest = _load_manifest(manifest_path)
    params = _load_params(config_path)

    sequences = _get_split_sequences(manifest, args.split)

    all_results = []
    strata_rows = []

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
                strata_rows=strata_rows,
                vote_window=args.vote_window,
                link_window=args.link_window,
                link_threshold=args.link_threshold,
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

    if strata_rows:
        strata_df = pd.DataFrame(strata_rows)
        strata_csv = output_dir / "strata.csv"
        strata_df.to_csv(strata_csv, index=False)
        print(f"\nĐã ghi: {strata_csv}")
        _print_strata(strata_df)
    else:
        print("\nKhông sequence nào có GT 2D -> không có bảng theo mức che.")


if __name__ == "__main__":
    main()