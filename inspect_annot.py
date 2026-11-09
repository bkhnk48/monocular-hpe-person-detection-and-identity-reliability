"""from pathlib import Path
import numpy as np
from scipy.io import loadmat

files = list(Path("data/raw/mpi_inf_3dhp").rglob("annot.mat"))

if not files:
    raise FileNotFoundError("Không tìm thấy annot.mat")

for path in files:
    print("\n" + "=" * 70)
    print("FILE:", path)

    data = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False
    )

    print("\nTOP-LEVEL VARIABLES:")
    for key, value in data.items():
        if not key.startswith("__"):
            print(key, type(value), getattr(value, "shape", None))

    for name in ["annot2", "annot3", "univ_annot3"]:
        if name not in data:
            print(f"\n{name}: NOT FOUND")
            continue

        arr = np.asarray(data[name])

        print("\n" + "-" * 50)
        print(name)
        print("Shape:", arr.shape)
        print("Dtype:", arr.dtype)

        if arr.dtype == object:
            print("Object array; first elements:")
            print(arr.flat[:3])
        else:
            print("First values:")
            print(arr[:3])

    break"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat

path = Path("data/raw/mpi_inf_3dhp/annot.mat")

data = loadmat(
    path,
    squeeze_me=True,
    struct_as_record=False
)

# 1. Kiểm tra số frame
frames = np.asarray(data["frames"])

print("=" * 70)
print("FRAMES")
print("Shape:", frames.shape)
print("First 10:", frames[:10])
print("Last 10:", frames[-10:])

# 2. Kiểm tra dữ liệu từng camera
for name in ["annot2", "annot3", "univ_annot3"]:
    print("\n" + "=" * 70)
    print(name)

    cameras_data = data[name]

    for cam_idx in [0, 1]:
        arr = np.asarray(cameras_data[cam_idx])

        print(f"\nCamera {cam_idx}")
        print("Shape:", arr.shape)
        print("Dtype:", arr.dtype)

        if arr.ndim != 2:
            continue

        print("First frame, raw:")
        print(arr[0])

        # Thử diễn giải thành các joint
        dims = 2 if name == "annot2" else 3

        if arr.shape[1] % dims == 0:
            joints = arr[0].reshape(-1, dims)
            print(f"\nFirst frame, reshaped into {dims}D joints:")
            for j, coord in enumerate(joints):
                print(f"  joint_{j:02d}: {coord}")

# 3. Kiểm tra metadata camera
print("\n" + "=" * 70)
print("CAMERA METADATA")

cameras = np.asarray(data["cameras"], dtype=object)

for cam_idx in [0, 1]:
    cam = cameras[cam_idx]

    print(f"\nCamera {cam_idx}")
    print("Type:", type(cam))

    fields = getattr(cam, "_fieldnames", None)

    if fields:
        for field in fields:
            value = getattr(cam, field)
            print(f"\n{field}:")
            print("  type:", type(value))
            print("  shape:", getattr(value, "shape", None))
            print("  value:", value)
    else:
        print(cam)
