import numpy as np
import pandas as pd
from scipy.io import loadmat

data = loadmat(
    "data/raw/mpi_inf_3dhp/annot.mat",
    squeeze_me=True,
    struct_as_record=False
)

frames = np.asarray(data["frames"]).reshape(-1)

gt2d = np.asarray(data["annot2"][0], dtype=np.float64)
gt3d = np.asarray(data["annot3"][0], dtype=np.float64)

gt2d = gt2d.reshape(len(frames), 28, 2)
gt3d = gt3d.reshape(len(frames), 28, 3)

fx, fy = 1497.693, 1497.103
cx, cy = 1024.704, 1051.394

X = gt3d[:, :, 0]
Y = gt3d[:, :, 1]
Z = gt3d[:, :, 2]

projected = np.stack([
    fx * X / Z + cx,
    fy * Y / Z + cy
], axis=2)

residual = projected - gt2d

rows = []

for f, frame_id in enumerate(frames):
    for j in [3, 18, 23]:
        rows.append({
            "frame": frame_id,
            "joint": j,

            "X3D": gt3d[f, j, 0],
            "Y3D": gt3d[f, j, 1],
            "Z3D": gt3d[f, j, 2],

            "u_GT": gt2d[f, j, 0],
            "v_GT": gt2d[f, j, 1],

            "u_projected": projected[f, j, 0],
            "v_projected": projected[f, j, 1],

            "residual_u": residual[f, j, 0],
            "residual_v": residual[f, j, 1],

            "error": np.linalg.norm(residual[f, j])
        })

df = pd.DataFrame(rows)

df.to_csv(
    "outputs/joint_reprojection_residuals.csv",
    index=False
)

print("Saved outputs/joint_reprojection_residuals.csv")

for j in [3, 18, 23]:
    subset = df[df["joint"] == j]

    print(f"\n=== JOINT {j} ===")

    for start, end in [(450, 550), (900, 1000)]:
        s = subset[
            (subset["frame"] >= start)
            & (subset["frame"] <= end)
        ]

        print(f"\nFrames {start}-{end}")
        print(
            s[[
                "frame",
                "X3D", "Y3D", "Z3D",
                "residual_u", "residual_v",
                "error"
            ]].iloc[::10].to_string(index=False)
        )
