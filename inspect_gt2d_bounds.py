import numpy as np
from scipy.io import loadmat

data = loadmat(
    "data/raw/mpi_inf_3dhp/annot.mat",
    squeeze_me=True,
    struct_as_record=False
)

frames = np.asarray(data["frames"]).reshape(-1)
gt2d = np.asarray(data["annot2"][0], dtype=np.float64)
gt2d = gt2d.reshape(len(frames), 28, 2)

# Nominal dimensions from calibration file
W, H = 2048, 2048

inside = (
    (gt2d[:, :, 0] >= 0)
    & (gt2d[:, :, 0] < W)
    & (gt2d[:, :, 1] >= 0)
    & (gt2d[:, :, 1] < H)
)

print("=== GLOBAL BOUNDS ===")
print("u min/max:", np.nanmin(gt2d[:, :, 0]), np.nanmax(gt2d[:, :, 0]))
print("v min/max:", np.nanmin(gt2d[:, :, 1]), np.nanmax(gt2d[:, :, 1]))

print("\n=== OUT-OF-BOUNDS COUNTS ===")
print("Total joints:", inside.size)
print("Inside:", inside.sum())
print("Outside:", (~inside).sum())

print("\n=== PER-JOINT ===")

for j in range(28):
    n_out = np.sum(~inside[:, j])
    print(f"Joint {j:02d}: outside={n_out:4d}/{len(frames)}")

print("\n=== SUSPICIOUS FRAMES ===")

for frame_id in [470, 480, 490, 500, 508, 510, 520, 529, 530, 540, 950]:
    f = np.where(frames == frame_id)[0][0]

    print(f"\nFrame {frame_id}")

    for j in [3, 4, 18, 19, 20, 23, 24, 25]:
        u, v = gt2d[f, j]

        print(
            f"Joint {j:02d}: "
            f"u={u:9.2f}, v={v:9.2f}, "
            f"inside={inside[f,j]}"
        )
