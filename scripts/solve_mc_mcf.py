import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

GT_FILE = Path(
    "data/raw/mpi_inf_3dhp/ground_truth/S8_Seq1_video_cam0.csv"
)

WHAM_FILE = Path(
    "data/raw/mpi_inf_3dhp/wham_predictions/RA-WHAM_S8_Seq1_video_cam0.csv"
)

OUT_DIR = Path("outputs/mc_mcf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# ------------------------------------------------------------
# Objective weights and penalties
# ------------------------------------------------------------

W_POSE = 1.0
W_TEMPORAL = 0.5

C_FP = 0.8
C_FN = 1.0

C_DISAPPEAR = 0.15
C_REAPPEAR = 0.20

# Reference scales for dimensionless costs
POSE_SCALE_MM = 100.0
MOTION_SCALE_M = 0.05

# Boundary costs: source -> first frame, last frame -> sink
C_BOUNDARY_START_DETECTED = 0.0
C_BOUNDARY_START_MISSED = 0.0
C_BOUNDARY_END_DETECTED = 0.0
C_BOUNDARY_END_MISSED = 0.0


# ============================================================
# 2. DATA UTILITIES
# ============================================================

def get_pose(row, joints):
    return np.array([
        [
            row[f"{joint}_x"],
            row[f"{joint}_y"],
            row[f"{joint}_z"]
        ]
        for joint in joints
    ], dtype=float)


def procrustes_align(pred, gt):
    """
    Align prediction to GT using:
    translation + rotation + uniform scale.

    Returns:
        PA-MPJPE in meters
        aligned prediction
        residual (aligned prediction - GT)
    """

    valid = (
        np.isfinite(pred).all(axis=1)
        & np.isfinite(gt).all(axis=1)
    )

    if valid.sum() < 8:
        return np.nan, None, None

    X = pred[valid]
    Y = gt[valid]

    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)

    Xc = X - mean_X
    Yc = Y - mean_Y

    norm_X = np.sum(Xc ** 2)

    if norm_X < 1e-12:
        return np.nan, None, None

    U, S, Vt = np.linalg.svd(Xc.T @ Yc)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt

    scale = np.sum(S) / norm_X

    aligned_valid = scale * (Xc @ R) + mean_Y

    errors = np.linalg.norm(aligned_valid - Y, axis=1)

    aligned = np.full_like(pred, np.nan)
    aligned[valid] = aligned_valid

    residual = np.full_like(pred, np.nan)
    residual[valid] = aligned_valid - Y

    return float(np.mean(errors)), aligned, residual


def temporal_motion_cost(state_a, state_b):
    """
    Temporal consistency cost for Detected -> Detected.

    Compare the change in pose-alignment residual between
    two consecutive frames.

    This does not assume that WHAM body_id remains unchanged.
    """

    ra = state_a["residual"]
    rb = state_b["residual"]

    valid = (
        np.isfinite(ra).all(axis=1)
        & np.isfinite(rb).all(axis=1)
    )

    if valid.sum() < 8:
        return 0.0

    delta = rb[valid] - ra[valid]

    rms = np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))

    return W_TEMPORAL * rms / MOTION_SCALE_M


# ============================================================
# 3. TRANSITION COSTS
# ============================================================

def transition_cost(a, b):

    if a["kind"] == "detected" and b["kind"] == "detected":
        return temporal_motion_cost(a, b)

    if a["kind"] == "detected" and b["kind"] == "missed":
        return C_DISAPPEAR

    if a["kind"] == "missed" and b["kind"] == "detected":
        return C_REAPPEAR

    # Missed -> Missed
    # FN is already charged in the unary cost.
    return 0.0


# ============================================================
# 4. BUILD FRAME STATES
# ============================================================

def build_states(gt, wham):

    gt_lookup = {
        int(row["frame"]): get_pose(row, JOINTS)
        for _, row in gt.iterrows()
    }

    wham_lookup = {
        frame: group
        for frame, group in wham.groupby("frame_id")
    }

    frames = sorted(gt_lookup.keys())

    all_states = []

    for frame in frames:

        gt_pose = gt_lookup[frame]

        group = wham_lookup.get(frame)

        states = []

        if group is not None:

            for _, row in group.iterrows():

                pred_pose = get_pose(row, JOINTS)

                pa_m, aligned, residual = procrustes_align(
                    pred_pose, gt_pose
                )

                if not np.isfinite(pa_m):
                    continue

                body_id = int(row["body_id"])

                # Unary pose cost.
                pose_cost = (
                    W_POSE * (pa_m * 1000.0) / POSE_SCALE_MM
                )

                states.append({
                    "kind": "detected",
                    "frame": frame,
                    "body_id": body_id,
                    "pa_mm": pa_m * 1000.0,
                    "residual": residual,
                    "pose_cost": pose_cost,
                })

        # Every frame has a Missed state.
        states.append({
            "kind": "missed",
            "frame": frame,
            "body_id": None,
            "pa_mm": np.nan,
            "residual": None,
            "pose_cost": 0.0,
        })

        n_detections = sum(
            s["kind"] == "detected" for s in states
        )

        # Exact unary objective:
        #
        # Detected:
        #   pose cost + FP penalty for all other detections
        #
        # Missed:
        #   FN penalty + FP penalty for every detection
        #
        # This is equivalent to:
        #   C_pose*x + C_FN*m + C_FP*sum(u_i)
        #
        # for one GT person per frame.

        for state in states:

            if state["kind"] == "detected":
                state["unary_cost"] = (
                    state["pose_cost"]
                    + C_FP * (n_detections - 1)
                )

            else:
                state["unary_cost"] = (
                    C_FN + C_FP * n_detections
                )

        all_states.append(states)

    return frames, all_states


# ============================================================
# 5. BOUNDARY COSTS
# ============================================================

def boundary_cost(state, is_start):

    if is_start:
        if state["kind"] == "detected":
            return C_BOUNDARY_START_DETECTED
        return C_BOUNDARY_START_MISSED

    if state["kind"] == "detected":
        return C_BOUNDARY_END_DETECTED

    return C_BOUNDARY_END_MISSED


# ============================================================
# 6. MIN-COST FLOW / SHORTEST PATH SOLVER
# ============================================================

def solve_shortest_path(frames, all_states):

    T = len(frames)

    if T == 0:
        raise RuntimeError("No GT frames found.")

    dp = []
    back = []

    # Initial frame: source -> state.
    first_costs = []

    for state in all_states[0]:
        cost = (
            state["unary_cost"]
            + boundary_cost(state, is_start=True)
        )
        first_costs.append(cost)

    dp.append(np.array(first_costs, dtype=float))
    back.append(np.full(len(first_costs), -1, dtype=int))

    # Intermediate frames.
    for t in range(1, T):

        previous_states = all_states[t - 1]
        current_states = all_states[t]

        current_costs = np.full(
            len(current_states), np.inf
        )

        current_back = np.full(
            len(current_states), -1, dtype=int
        )

        for b_idx, b in enumerate(current_states):

            best_cost = np.inf
            best_prev = -1

            for a_idx, a in enumerate(previous_states):

                cost = (
                    dp[t - 1][a_idx]
                    + transition_cost(a, b)
                    + b["unary_cost"]
                )

                if cost < best_cost:
                    best_cost = cost
                    best_prev = a_idx

            current_costs[b_idx] = best_cost
            current_back[b_idx] = best_prev

        dp.append(current_costs)
        back.append(current_back)

    # Final frame: state -> sink.
    final_costs = np.array([
        dp[-1][i] + boundary_cost(state, is_start=False)
        for i, state in enumerate(all_states[-1])
    ])

    best_final_state = int(np.argmin(final_costs))
    optimal_objective = float(final_costs[best_final_state])

    # Backtracking.
    selected_indices = [None] * T
    selected_indices[-1] = best_final_state

    for t in range(T - 1, 0, -1):
        selected_indices[t - 1] = int(
            back[t][selected_indices[t]]
        )

    selected_states = [
        all_states[t][selected_indices[t]]
        for t in range(T)
    ]

    return selected_states, optimal_objective


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("Loading GT:", GT_FILE)
    print("Loading WHAM:", WHAM_FILE)

    gt = pd.read_csv(GT_FILE)

    # This GT CSV contains all 14 cameras.
    # Select camera 0 to match the WHAM cam0 sequence.
    gt["camera"] = pd.to_numeric(
        gt["camera"], errors="coerce"
    )
    gt = gt[gt["camera"] == 0].copy()

    if gt.empty:
        raise ValueError("No GT rows found for camera 0.")

    if gt["frame"].duplicated().any():
        raise ValueError(
            "GT still contains duplicate frame IDs after camera filtering."
        )

    print("Selected GT camera: 0")
    print("GT rows after camera filtering:", len(gt))

    wham = pd.read_csv(WHAM_FILE)

    gt["frame"] = pd.to_numeric(
        gt["frame"], errors="coerce"
    )

    wham["frame_id"] = pd.to_numeric(
        wham["frame_id"], errors="coerce"
    )

    wham["body_id"] = pd.to_numeric(
        wham["body_id"], errors="coerce"
    )

    gt = gt.dropna(subset=["frame"]).copy()
    wham = wham.dropna(
        subset=["frame_id", "body_id"]
    ).copy()

    gt["frame"] = gt["frame"].astype(int)
    wham["frame_id"] = wham["frame_id"].astype(int)
    wham["body_id"] = wham["body_id"].astype(int)

    # GT CSV is in mm. WHAM CSV is in meters.
    for joint in JOINTS:
        for axis in ["x", "y", "z"]:
            col = f"{joint}_{axis}"

            gt[col] = (
                pd.to_numeric(gt[col], errors="coerce")
                / 1000.0
            )

            wham[col] = pd.to_numeric(
                wham[col], errors="coerce"
            )

    print("GT frames:", len(gt))
    print("WHAM detections:", len(wham))
    print("WHAM body IDs:", sorted(wham["body_id"].unique()))

    frames, all_states = build_states(gt, wham)

    print("\nBuilding temporal flow network...")
    print("Number of frame layers:", len(frames))
    print(
        "Number of detection states:",
        sum(
            s["kind"] == "detected"
            for states in all_states
            for s in states
        )
    )

    print("\nSolving min-cost flow...")
    selected, objective = solve_shortest_path(
        frames, all_states
    )

    # --------------------------------------------------------
    # Reconstruct assignment and objective components.
    # --------------------------------------------------------

    wham_lookup = {
        frame: group
        for frame, group in wham.groupby("frame_id")
    }

    records = []

    for t, (frame, state) in enumerate(zip(frames, selected)):

        group = wham_lookup.get(frame)

        if group is None:
            detected_ids = []
        else:
            detected_ids = sorted(
                group["body_id"].astype(int).tolist()
            )

        if state["kind"] == "detected":
            selected_id = state["body_id"]

            unmatched_ids = [
                x for x in detected_ids if x != selected_id
            ]

            status = "detected"
            selected_pa = state["pa_mm"]

        else:
            selected_id = np.nan
            unmatched_ids = detected_ids

            status = "missed"
            selected_pa = np.nan

        if t == 0:
            trans_cost = 0.0
        else:
            trans_cost = transition_cost(
                selected[t - 1], state
            )

        records.append({
            "frame": frame,
            "status": status,
            "selected_body_id": selected_id,
            "selected_pa_mpjpe_mm": selected_pa,
            "number_of_detections": len(detected_ids),
            "all_detected_body_ids": ",".join(
                map(str, detected_ids)
            ),
            "unmatched_body_ids": ",".join(
                map(str, unmatched_ids)
            ),
            "unary_cost": state["unary_cost"],
            "transition_cost": trans_cost,
            "boundary_cost": boundary_cost(
                state, is_start=(t == 0)
            ) if t == 0 else 0.0,
        })

    result = pd.DataFrame(records)

    result.to_csv(
        OUT_DIR / "mc_mcf_assignment.csv",
        index=False
    )

    # --------------------------------------------------------
    # Body ID statistics after global optimization.
    # --------------------------------------------------------

    matched = result[
        result["status"] == "detected"
    ].copy()

    id_summary = (
        matched.groupby("selected_body_id")
        .agg(
            matched_frames=("frame", "count"),
            median_pa_mpjpe_mm=(
                "selected_pa_mpjpe_mm", "median"
            ),
            mean_pa_mpjpe_mm=(
                "selected_pa_mpjpe_mm", "mean"
            ),
        )
        .reset_index()
        .sort_values(
            "matched_frames", ascending=False
        )
    )

    id_summary.to_csv(
        OUT_DIR / "mc_mcf_body_id_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Objective breakdown.
    # --------------------------------------------------------

    pose_total = (
        matched["selected_pa_mpjpe_mm"].sum()
        / POSE_SCALE_MM
        * W_POSE
    )

    fp_total = sum(
        len(row["unmatched_body_ids"].split(","))
        for _, row in result.iterrows()
        if row["unmatched_body_ids"]
    ) * C_FP

    fn_total = (
        (result["status"] == "missed").sum()
        * C_FN
    )

    temporal_total = result["transition_cost"].sum()

    boundary_total = (
        result["boundary_cost"].sum()
        + (
            C_BOUNDARY_END_MISSED
            if selected[-1]["kind"] == "missed"
            else C_BOUNDARY_END_DETECTED
        )
    )

    print("\n" + "=" * 65)
    print("MC-MCF OPTIMIZATION RESULTS")
    print("=" * 65)

    print(f"Optimal objective: {objective:.6f}")

    print("\nObjective breakdown:")
    print(f"Pose unary cost:       {pose_total:.6f}")
    print(f"False positive cost:   {fp_total:.6f}")
    print(f"False negative cost:   {fn_total:.6f}")
    print(f"Temporal cost:         {temporal_total:.6f}")
    print(f"Boundary cost:         {boundary_total:.6f}")

    print("\nAssignment statistics:")
    print("Total GT frames:", len(result))
    print("Matched frames:", len(matched))
    print(
        "Missed frames:",
        int((result["status"] == "missed").sum())
    )

    print("\nSelected body_id statistics:")
    if len(id_summary):
        print(id_summary.to_string(index=False))
    else:
        print("No detections selected.")

    print("\nOutput files:")
    print(OUT_DIR / "mc_mcf_assignment.csv")
    print(OUT_DIR / "mc_mcf_body_id_summary.csv")


if __name__ == "__main__":
    main()
