import numpy as np
from scipy.io import loadmat
from pathlib import Path

mat_path = Path("data/raw/mpi_inf_3dhp/annot.mat")

fx, fy = 1497.693, 1497.103
cx, cy = 1024.704, 1051.394

data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

frames = np.asarray(data["frames"]).reshape(-1)
gt2d = np.asarray(data["annot2"][0], dtype=np.float64)
gt3d = np.asarray(data["annot3"][0], dtype=np.float64)

gt2d = gt2d.reshape(len(frames), 28, 2)
gt3d = gt3d.reshape(len(frames), 28, 3)

X = gt3d[:, :, 0]
Y = gt3d[:, :, 1]
Z = gt3d[:, :, 2]

pred2d = np.stack([
    fx * X / Z + cx,
    fy * Y / Z + cy
], axis=2)

errors = np.linalg.norm(pred2d - gt2d, axis=2)

valid = (
    np.isfinite(gt3d).all(axis=2)
    & np.isfinite(gt2d).all(axis=2)
    & (Z > 0)
)

errors[~valid] = np.nan

# --------------------------------------------------
# 1. Largest individual errors
# --------------------------------------------------

print("\n=== TOP 30 REPROJECTION ERRORS ===")

flat_indices = np.argsort(
    np.nan_to_num(errors, nan=-1).ravel()
)[::-1][:30]

for idx in flat_indices:
    f, j = np.unravel_index(idx, errors.shape)

    print(
        f"frame={frames[f]:4d}, "
        f"joint={j:02d}, "
        f"error={errors[f,j]:10.4f}px, "
        f"GT2D={gt2d[f,j]}, "
        f"Projected={pred2d[f,j]}"
    )

# --------------------------------------------------
# 2. Error distribution per joint
# --------------------------------------------------

print("\n=== ERROR COUNTS PER JOINT ===")

for j in range(28):
    e = errors[:, j]
    e = e[np.isfinite(e)]

    print(
        f"Joint {j:02d}: "
        f">1px={np.sum(e > 1):4d}, "
        f">3px={np.sum(e > 3):4d}, "
        f">5px={np.sum(e > 5):4d}, "
        f">10px={np.sum(e > 10):4d}, "
        f"max={np.max(e):9.3f}px"
    )

# --------------------------------------------------
# 3. Inspect three suspicious joints
# --------------------------------------------------

for j in [3, 18, 23]:
    e = errors[:, j]
    worst = np.argsort(
        np.nan_to_num(e, nan=-1)
    )[::-1][:10]

    print(f"\n=== JOINT {j:02d}: TOP 10 FRAMES ===")

    for f in worst:
        print(
            f"frame={frames[f]:4d}, "
            f"error={e[f]:9.4f}px, "
            f"GT2D={gt2d[f,j]}, "
            f"Projected={pred2d[f,j]}"
        )

# --------------------------------------------------
# 4. Frames with unusually large errors
# --------------------------------------------------

frame_max = np.nanmax(errors, axis=1)

print("\n=== FRAMES WITH MAX ERROR > 10 PX ===")

for f in np.where(frame_max > 10)[0]:
    print(
        f"frame={frames[f]:4d}, "
        f"max_error={frame_max[f]:9.3f}px, "
        f"joint={np.nanargmax(errors[f]):02d}"
    )
