
from pathlib import Path
from itertools import combinations

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

WHAM_PATH = (
    ROOT
    / "data/raw/mpi_inf_3dhp/wham_predictions/"
    / "RA-WHAM_S8_Seq1_video_cam0.csv"
)

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(WHAM_PATH)

# One row per frame, with all WHAM body IDs.
frame_ids = (
    df.groupby("frame_id")["body_id"]
    .apply(lambda x: sorted(set(x.astype(int))))
    .reset_index(name="body_ids")
)

# Per-body-ID statistics.
rows = []

for body_id, group in df.groupby("body_id"):
    frames = sorted(group["frame_id"].astype(int).unique())

    rows.append({
        "body_id": int(body_id),
        "detection_rows": len(group),
        "unique_frames": len(frames),
        "first_frame": min(frames),
        "last_frame": max(frames),
        "frame_coverage": len(frames) / 6468,
    })

body_summary = pd.DataFrame(rows).sort_values(
    "unique_frames", ascending=False
)

body_summary.to_csv(
    OUT_DIR / "wham_body_id_summary.csv",
    index=False,
)

# Count pairs of IDs co-occurring in the same frame.
pair_counts = {}

for ids in frame_ids["body_ids"]:
    for a, b in combinations(ids, 2):
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

pair_rows = [
    {
        "body_id_a": a,
        "body_id_b": b,
        "cooccurring_frames": count,
    }
    for (a, b), count in pair_counts.items()
]

pair_summary = pd.DataFrame(
    pair_rows,
    columns=[
        "body_id_a",
        "body_id_b",
        "cooccurring_frames",
    ],
).sort_values(
    "cooccurring_frames",
    ascending=False,
)

pair_summary.to_csv(
    OUT_DIR / "wham_body_id_cooccurrence.csv",
    index=False,
)

# Consecutive-frame persistence for each body ID.
persistence_rows = []

for body_id in sorted(df["body_id"].unique()):
    frames = sorted(
        df.loc[
            df["body_id"] == body_id,
            "frame_id"
        ].astype(int).unique()
    )

    consecutive = sum(
        b == a + 1 for a, b in zip(frames, frames[1:])
    )

    persistence_rows.append({
        "body_id": int(body_id),
        "unique_frames": len(frames),
        "consecutive_frame_pairs": consecutive,
        "consecutive_ratio": (
            consecutive / (len(frames) - 1)
            if len(frames) > 1 else 0
        ),
    })

persistence = pd.DataFrame(persistence_rows)

persistence.to_csv(
    OUT_DIR / "wham_body_id_persistence.csv",
    index=False,
)

print("\n=== BODY ID SUMMARY ===")
print(body_summary.to_string(index=False))

print("\n=== BODY ID CO-OCCURRENCE ===")
print(pair_summary.to_string(index=False))

print("\n=== TEMPORAL PERSISTENCE ===")
print(persistence.to_string(index=False))

print("\nSaved:")
print("outputs/wham_body_id_summary.csv")
print("outputs/wham_body_id_cooccurrence.csv")
print("outputs/wham_body_id_persistence.csv")
