from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "outputs/subject_matching/best_detection_per_frame.csv"
)

OUTPUT_DIR = Path("outputs/subject_matching")

SEGMENTS_PATH = OUTPUT_DIR / "best_bodyid_segments.csv"
TRANSITIONS_PATH = OUTPUT_DIR / "best_bodyid_transitions.csv"


def main():
    print("=" * 70)
    print("ANALYZE BEST BODY ID TRANSITIONS")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    df = df.sort_values("frame_id").reset_index(drop=True)

    df["frame_id"] = df["frame_id"].astype(int)

    # --------------------------------------------------------
    # 1. Create a label for each frame
    # --------------------------------------------------------

    def get_label(row):
        if pd.isna(row["best_body_id"]):
            return "NO_DETECTION"

        return str(int(row["best_body_id"]))

    df["best_label"] = df.apply(get_label, axis=1)

    # --------------------------------------------------------
    # 2. Detect contiguous runs of the same winning ID
    # --------------------------------------------------------

    df["new_segment"] = (
        df["best_label"] != df["best_label"].shift()
    )

    df["segment_id"] = df["new_segment"].cumsum()

    segment_records = []

    for segment_id, group in df.groupby("segment_id"):

        first = group.iloc[0]
        last = group.iloc[-1]

        label = first["best_label"]

        if label == "NO_DETECTION":
            body_id = None
        else:
            body_id = int(label)

        scores = group["best_pa_mpjpe_mm"].dropna()

        if len(scores) > 0:
            median_error = float(scores.median())
            mean_error = float(scores.mean())
        else:
            median_error = None
            mean_error = None

        segment_records.append({
            "segment_id": int(segment_id),
            "best_body_id": body_id,
            "segment_type": label,
            "start_frame": int(first["frame_id"]),
            "end_frame": int(last["frame_id"]),
            "number_of_frames": int(len(group)),
            "median_best_pa_mpjpe_mm": median_error,
            "mean_best_pa_mpjpe_mm": mean_error,
            "mean_detection_count": float(
                group["detection_count"].mean()
            ),
        })

    segments = pd.DataFrame(segment_records)

    segments.to_csv(SEGMENTS_PATH, index=False)

    print("\nSaved:", SEGMENTS_PATH)
    print("Number of winning-ID segments:", len(segments))

    # --------------------------------------------------------
    # 3. Create transitions between consecutive segments
    # --------------------------------------------------------

    transition_records = []

    for i in range(1, len(segments)):

        previous = segments.iloc[i - 1]
        current = segments.iloc[i]

        transition_records.append({
            "transition_id": i,
            "from_body_id": previous["best_body_id"],
            "to_body_id": current["best_body_id"],
            "from_type": previous["segment_type"],
            "to_type": current["segment_type"],
            "previous_end_frame": int(previous["end_frame"]),
            "next_start_frame": int(current["start_frame"]),
            "transition_frame": int(current["start_frame"]),
            "frames_between": int(
                current["start_frame"]
                - previous["end_frame"]
                - 1
            ),
            "previous_segment_length": int(
                previous["number_of_frames"]
            ),
            "next_segment_length": int(
                current["number_of_frames"]
            ),
        })

    transitions = pd.DataFrame(transition_records)

    transitions.to_csv(TRANSITIONS_PATH, index=False)

    print("\nSaved:", TRANSITIONS_PATH)
    print("Number of transitions:", len(transitions))

    # --------------------------------------------------------
    # 4. Print winning-ID timeline
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WINNING BODY ID TIMELINE")
    print("=" * 70)

    print(
        segments[
            [
                "segment_id",
                "best_body_id",
                "start_frame",
                "end_frame",
                "number_of_frames",
                "median_best_pa_mpjpe_mm",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. Print transitions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BODY ID TRANSITIONS")
    print("=" * 70)

    if not transitions.empty:
        print(
            transitions[
                [
                    "from_type",
                    "to_type",
                    "previous_end_frame",
                    "next_start_frame",
                    "frames_between",
                    "previous_segment_length",
                    "next_segment_length",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("Total frames:", len(df))
    print(
        "Frames without detection:",
        int((df["best_label"] == "NO_DETECTION").sum())
    )

    print("\nWinning ID frame counts:")

    print(
        df["best_label"]
        .value_counts()
        .to_string()
    )

    print("\nLongest winning-ID segments:")

    print(
        segments.sort_values(
            "number_of_frames",
            ascending=False
        ).head(20)[
            [
                "best_body_id",
                "start_frame",
                "end_frame",
                "number_of_frames",
                "median_best_pa_mpjpe_mm",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
