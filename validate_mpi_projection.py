import numpy as np
from scipy.io import loadmat
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

mat_path = Path("data/raw/mpi_inf_3dhp/annot.mat")

# Camera 0 intrinsic
fx = 1497.693
fy = 1497.103
cx = 1024.704
cy = 1051.394

# --------------------------------------------------
# Load annotations
# --------------------------------------------------

data = loadmat(
    mat_path,
    squeeze_me=True,
    struct_as_record=False
)

frames = np.asarray(data["frames"]).reshape(-1)

# Camera 0
gt2d = np.asarray(data["annot2"][0], dtype=np.float64)
gt3d = np.asarray(data["annot3"][0], dtype=np.float64)

print("Frames:", len(frames))
print("GT 2D:", gt2d.shape)
print("GT 3D:", gt3d.shape)

# 28 joints
gt2d = gt2d.reshape(len(frames), 28, 2)
gt3d = gt3d.reshape(len(frames), 28, 3)

# --------------------------------------------------
# Project 3D -> 2D
# --------------------------------------------------

X = gt3d[:, :, 0]
Y = gt3d[:, :, 1]
Z = gt3d[:, :, 2]

valid = (
    np.isfinite(gt3d).all(axis=2)
    & np.isfinite(gt2d).all(axis=2)
    & (Z > 0)
)

pred2d = np.full_like(gt2d, np.nan)

pred2d[:, :, 0] = fx * X / Z + cx
pred2d[:, :, 1] = fy * Y / Z + cy

# --------------------------------------------------
# Reprojection error
# --------------------------------------------------

errors = np.linalg.norm(pred2d - gt2d, axis=2)

valid_errors = errors[valid]

print("\n=== REPROJECTION ERROR ===")
print("Valid observations:", len(valid_errors))
print("Total observations:", errors.size)

print("Mean error:", np.mean(valid_errors), "pixels")
print("Median error:", np.median(valid_errors), "pixels")
print("P95 error:", np.percentile(valid_errors, 95), "pixels")
print("Maximum error:", np.max(valid_errors), "pixels")

# --------------------------------------------------
# Per-joint error
# --------------------------------------------------

print("\n=== PER-JOINT ERROR ===")

for j in range(28):
    mask = valid[:, j]

    if not np.any(mask):
        continue

    e = errors[mask, j]

    print(
        f"Joint {j:02d}: "
        f"mean={np.mean(e):.6f}, "
        f"median={np.median(e):.6f}, "
        f"p95={np.percentile(e, 95):.6f}"
    )

# --------------------------------------------------
# First frame diagnostic
# --------------------------------------------------

print("\n=== FIRST FRAME ===")

for j in range(28):
    print(
        f"Joint {j:02d}: "
        f"GT2D={gt2d[0, j]}, "
        f"Projected={pred2d[0, j]}, "
        f"Error={errors[0, j]:.6f}"
    )