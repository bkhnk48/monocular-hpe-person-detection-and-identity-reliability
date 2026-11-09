
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]

ANNOT_PATH = ROOT / "data/raw/mpi_inf_3dhp/annot.mat"
OUTPUT_PATH = (
    ROOT
    / "data/processed/mpi_inf_3dhp/gt_frame_status.csv"
)

IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 2048

# Camera 0
CAMERA_ID = 0

data = loadmat(
    ANNOT_PATH,
    squeeze_me=True,
    struct_as_record=False,
)

frames = np.asarray(data["frames"]).reshape(-1)

gt2d = np.asarray(
    data["annot2"][CAMERA_ID],
    dtype=np.float64,
).reshape(len(frames), 28, 2)

gt3d = np.asarray(
    data["annot3"][CAMERA_ID],
    dtype=np.float64,
).reshape(len(frames), 28, 3)


# A 2D joint is inside the image if its coordinates
# are finite and within image boundaries.
valid_2d = np.isfinite(gt2d).all(axis=2)

inside_image = (
    valid_2d
    & (gt2d[:, :, 0] >= 0)
    & (gt2d[:, :, 0] < IMAGE_WIDTH)
    & (gt2d[:, :, 1] >= 0)
    & (gt2d[:, :, 1] < IMAGE_HEIGHT)
)

valid_3d = np.isfinite(gt3d).all(axis=2)

visible_joint_count = inside_image.sum(axis=1)
valid_2d_count = valid_2d.sum(axis=1)
valid_3d_count = valid_3d.sum(axis=1)

visible_joint_ratio = visible_joint_count / gt2d.shape[1]


df = pd.DataFrame({
    "frame_id": frames,

    "person_present": pd.Series(
        [pd.NA] * len(frames), dtype="boolean"
    ),

    "visibility_level": "unreviewed",

    "visible_joint_count": visible_joint_count,

    "visible_joint_ratio": visible_joint_ratio,

    "gt3d_available": valid_3d_count > 0,

    "gt2d_available": valid_2d_count > 0,

    "valid_3d_joint_count": valid_3d_count,

    "valid_2d_joint_count": valid_2d_count,

    "manual_person_present": pd.Series(
        [pd.NA] * len(frames), dtype="boolean"
    ),

    "manual_visibility_level": "unreviewed",

    "annotation_status": "unreviewed",

    "review_notes": "",
})


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Frames:", len(df))

print("\nVisible joint count distribution:")
print(df["visible_joint_count"].value_counts().sort_index())

print("\nGT availability:")
print("GT2D available:", df["gt2d_available"].sum())
print("GT3D available:", df["gt3d_available"].sum())
