# ruff: noqa: E402
"""Import head pose from OpenFace
=================================

Extract head pose rotations from OpenFace output .csv file.
"""

# %%
# Imports
# -------

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

# %%
# Define paths to the OpenFace output and load the data
# ------------------------------------------------------

data_dir = Path.home() / "Data" / "CervicalDystonia_OpenFace"
assert data_dir.exists(), f"Path not found: {data_dir}"

video_name = "C12c192_rest.mp4"
video_path = data_dir / video_name

open_face_dir = data_dir / "OpenFace"
csv_name = video_name.replace(".mp4", ".csv")
csv_path = open_face_dir / csv_name
assert csv_path.exists(), f"Path not found: {csv_path}"

# %%
# Load csv with pandas

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # strip whitespace from column names
df.head()

# %%
# Keep only necessary columns

keep_columns = [
    "frame",
    "timestamp",
    "confidence",
    "success",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
]
df_keep = df[keep_columns]
df_keep.head()

# %%
# Plot the head pose rotations over time
# ---------------------------------------

for col in ["pose_Rx", "pose_Ry", "pose_Rz"]:
    plt.plot(df_keep["frame"], df_keep[col], label=col)
plt.legend()

# %%
# Let's put the head rotations into an xarray DataArray

import xarray as xr

# %%
# Define the time dimension

time = df_keep["timestamp"].values

# Define a "rotation_axis" dimension with coords "x", "y", "z"
rotation_axis = ["x", "y", "z"]

# Define the data array
rotations = xr.DataArray(
    df_keep[["pose_Rx", "pose_Ry", "pose_Rz"]].values,
    dims=("time", "rotation_axis"),
    coords={"time": time, "rotation_axis": rotation_axis},
    name="head_pose_rotations",
)
# %%
rotations.plot.line(x="time", hue="rotation_axis")

# %%
# Construct 3x3 rotation matrices from the Euler angles

from scipy.spatial.transform import Rotation

R = Rotation.from_euler("xyz", rotations.values, degrees=False).as_matrix()

print(R.shape)
print(R[0])

# %%
# Get 60-th frame to use as a neutral head pose

neutral_frame = 60
neutral_R = R[neutral_frame]  # Get the rotation matrix of neutral head pose
print(neutral_R)

# %%
# Decompose the rotation matrix at each frame into two components:
# 1. A rotation matrix from camera coordinates to neutral head pose
# 2. A rotation matrix from neutral head pose to head pose at each frame


# %%
# Convert the 2nd matrix to Euler angles
# This time they will be in relation to the neutral head pose
