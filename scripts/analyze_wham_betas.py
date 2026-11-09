
#!/usr/bin/env python3

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def robust_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = robust_mad(x)

    if not np.isfinite(mad) or mad < 1e-12:
        std = np.nanstd(x)
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(x)
        return (x - med) / std

    return (x - med) / mad

import joblib


def load_bodies(pkl_path):
    data = joblib.load(pkl_path)

    if not isinstance(data, dict) or not data:
        raise ValueError(
            "File WHAM không chứa dictionary subject/track hợp lệ."
        )

    bodies = {
        body_id: body
        for body_id, body in data.items()
        if isinstance(body, dict)
        and "frame_ids" in body
        and "betas" in body
    }

    if not bodies:
        print("Top-level keys:", list(data.keys())[:30])
        raise ValueError(
            "Không tìm thấy Body ID có frame_ids và betas."
        )

    return bodies

def load_arrays(body):
    frames = np.asarray(body["frame_ids"]).reshape(-1).astype(int)
    betas = np.asarray(body["betas"], dtype=float)

    if betas.ndim == 1:
        betas = betas.reshape(1, -1)

    if len(frames) != len(betas):
        raise ValueError("frame_ids and betas have different lengths")

    trans = body.get("trans")
    if trans is not None:
        trans = np.asarray(trans, dtype=float)
        if trans.ndim == 1:
            trans = trans.reshape(1, -1)
        if len(trans) != len(frames):
            trans = None

    order = np.argsort(frames)
    frames = frames[order]
    betas = betas[order]
    if trans is not None:
        trans = trans[order]

    return frames, betas, trans


def dist(a, b):
    return float(np.linalg.norm(a - b))


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--abnormal-frames", default="445")
    parser.add_argument("--jump-mad-threshold", type=float, default=6.0)
    args = parser.parse_args()

    abnormal_manual = {
        int(x.strip())
        for x in args.abnormal_frames.split(",")
        if x.strip()
    }

    print("Loading:", args.pkl)
    bodies_raw = load_bodies(args.pkl)

    records = {}

    for body_id, body in bodies_raw.items():
        frames, betas, trans = load_arrays(body)

        beta_step = np.full(len(frames), np.nan)
        if len(frames) > 1:
            beta_step[1:] = np.linalg.norm(np.diff(betas, axis=0), axis=1)

        trans_step = np.full(len(frames), np.nan)
        if trans is not None and len(frames) > 1:
            trans_step[1:] = np.linalg.norm(np.diff(trans, axis=0), axis=1)

        records[body_id] = {
            "frames": frames,
            "betas": betas,
            "trans": trans,
            "beta_step": beta_step,
            "trans_step": trans_step,
        }

    print(f"\nNumber of Body IDs: {len(records)}")

    # ---------------------------------------------------------
    # Detect possible anomalies using trans jumps.
    # ---------------------------------------------------------

    abnormal_auto = set()

    for body_id, rec in records.items():
        jumps = rec["trans_step"]
        valid = np.isfinite(jumps)

        if valid.sum() < 5:
            continue

        z = robust_zscore(jumps[valid])
        indices = np.where(valid)[0][z > args.jump_mad_threshold]

        for idx in indices:
            abnormal_auto.add(int(rec["frames"][idx]))

    abnormal = abnormal_manual | abnormal_auto

    print("\nManually specified abnormal frames:", sorted(abnormal_manual))
    print("Automatically flagged frames:", sorted(abnormal_auto))
    print("Combined abnormal frames:", sorted(abnormal))

    # ---------------------------------------------------------
    # Normal beta prototypes.
    # ---------------------------------------------------------

    prototypes = {}

    for body_id, rec in records.items():
        mask = np.array([
            int(f) not in abnormal
            for f in rec["frames"]
        ])

        if not mask.any():
            mask = np.ones(len(rec["frames"]), dtype=bool)

        prototypes[body_id] = np.median(rec["betas"][mask], axis=0)

    # ---------------------------------------------------------
    # 1. Beta variability within each Body ID.
    # ---------------------------------------------------------

    section("1. BETA VARIABILITY WITHIN EACH BODY ID")

    rows = []

    for body_id, rec in records.items():
        betas = rec["betas"]
        prototype = prototypes[body_id]

        distances = np.linalg.norm(
            betas - prototype[None, :], axis=1
        )

        rows.append({
            "Body_ID": str(body_id),
            "Frames": len(betas),
            "Beta_dim": betas.shape[1],
            "Beta_std_L2": np.linalg.norm(np.std(betas, axis=0)),
            "Median_dist_to_prototype": np.median(distances),
            "P95_dist_to_prototype": np.percentile(distances, 95),
            "Max_dist_to_prototype": np.max(distances),
            "Median_frame_to_frame_change": np.nanmedian(rec["beta_step"]),
            "P95_frame_to_frame_change": np.nanpercentile(
                rec["beta_step"][np.isfinite(rec["beta_step"])], 95
            ) if np.isfinite(rec["beta_step"]).any() else np.nan,
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ---------------------------------------------------------
    # 2. Pairwise distances between Body IDs.
    # ---------------------------------------------------------

    section("2. PAIRWISE BETA DISTANCES BETWEEN BODY IDs")

    ids = list(prototypes.keys())
    matrix = np.zeros((len(ids), len(ids)))

    for i, id_a in enumerate(ids):
        for j, id_b in enumerate(ids):
            matrix[i, j] = dist(
                prototypes[id_a], prototypes[id_b]
            )

    matrix_df = pd.DataFrame(
        matrix,
        index=[str(x) for x in ids],
        columns=[str(x) for x in ids],
    )

    print("\nEuclidean distance matrix (normal-frame beta prototypes):")
    print(matrix_df.to_string(float_format=lambda x: f"{x:.6f}"))

    pair_rows = []

    for i, j in itertools.combinations(range(len(ids)), 2):
        a, b = ids[i], ids[j]
        pair_rows.append({
            "Body_A": str(a),
            "Body_B": str(b),
            "Euclidean": matrix[i, j],
        })

    if pair_rows:
        pair_df = pd.DataFrame(pair_rows).sort_values("Euclidean")
        print("\nClosest Body ID pairs:")
        print(pair_df.head(15).to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        ))

    # ---------------------------------------------------------
    # 3. Normal vs abnormal frames.
    # ---------------------------------------------------------

    section("3. NORMAL VS ABNORMAL FRAME COMPARISON")

    comparison_rows = []

    for body_id, rec in records.items():
        frames = rec["frames"]
        betas = rec["betas"]

        normal_mask = np.array([int(f) not in abnormal for f in frames])
        abnormal_mask = ~normal_mask

        prototype = prototypes[body_id]
        beta_dist = np.linalg.norm(
            betas - prototype[None, :], axis=1
        )

        for metric_name, values in [
            ("distance_to_normal_prototype", beta_dist),
            ("frame_to_frame_beta_change", rec["beta_step"]),
            ("translation_jump", rec["trans_step"]),
        ]:
            normal_values = values[normal_mask]
            abnormal_values = values[abnormal_mask]

            normal_values = normal_values[np.isfinite(normal_values)]
            abnormal_values = abnormal_values[np.isfinite(abnormal_values)]

            comparison_rows.append({
                "Body_ID": str(body_id),
                "Metric": metric_name,
                "N_normal": len(normal_values),
                "N_abnormal": len(abnormal_values),
                "Normal_median": np.median(normal_values)
                    if len(normal_values) else np.nan,
                "Normal_P95": np.percentile(normal_values, 95)
                    if len(normal_values) else np.nan,
                "Abnormal_median": np.median(abnormal_values)
                    if len(abnormal_values) else np.nan,
                "Abnormal_max": np.max(abnormal_values)
                    if len(abnormal_values) else np.nan,
            })

    comparison = pd.DataFrame(comparison_rows)
    print(comparison.to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))

    # ---------------------------------------------------------
    # 4. Detailed frame 445 analysis.
    # ---------------------------------------------------------

    section("4. FRAME 445: BETA COMPARISON")

    target_frame = 445
    found = []

    for body_id, rec in records.items():
        positions = np.where(rec["frames"] == target_frame)[0]

        if len(positions) == 0:
            continue

        idx = int(positions[0])
        beta = rec["betas"][idx]

        found.append((body_id, beta, rec, idx))

    if not found:
        print("No Body ID has a detection at frame 445.")
    else:
        for body_id, beta, rec, idx in found:
            print(f"\nBody ID: {body_id}")
            print("Beta at frame 445:")
            print(np.array2string(beta, precision=6))

            print(
                "Distance to own normal prototype:",
                f"{dist(beta, prototypes[body_id]):.6f}"
            )

            print(
                "Beta frame-to-frame change:",
                f"{rec['beta_step'][idx]:.6f}"
            )

            print(
                "Translation jump:",
                f"{rec['trans_step'][idx]:.6f}"
            )

            distances = sorted(
                [
                    (other_id, dist(beta, prototype))
                    for other_id, prototype in prototypes.items()
                ],
                key=lambda x: x[1],
            )

            print("\nDistance to normal prototypes of all Body IDs:")
            for other_id, distance in distances:
                print(f"  Body {other_id}: {distance:.6f}")

    # ---------------------------------------------------------
    # 5. Preliminary interpretation indicators.
    # ---------------------------------------------------------

    section("5. PRELIMINARY DISCRIMINABILITY INDICATORS")

    within_medians = summary["Median_dist_to_prototype"].to_numpy()
    within_p95 = summary["P95_dist_to_prototype"].to_numpy()

    between = matrix[np.triu_indices(len(ids), k=1)]

    if len(between):
        print("Median within-Body beta distance:",
              f"{np.median(within_medians):.6f}")
        print("Median within-Body P95 distance:",
              f"{np.median(within_p95):.6f}")
        print("Median between-Body prototype distance:",
              f"{np.median(between):.6f}")
        print("Minimum between-Body prototype distance:",
              f"{np.min(between):.6f}")
        print("Maximum between-Body prototype distance:",
              f"{np.max(between):.6f}")

        ratio = np.median(between) / max(
            np.median(within_medians), 1e-12
        )

        print("Between/within median-distance ratio:",
              f"{ratio:.6f}")

    print("\nAnalysis completed. No CSV files were created.")


if __name__ == "__main__":
    main()

