# ruff: noqa: E402
"""Compute and visualise kinematics.
====================================

Compute displacement, velocity and acceleration, and
visualise the results.
"""

# %%
# Imports
# -------

import numpy as np
from matplotlib import pyplot as plt

from movement import sample_data
from movement.plots import plot_occupancy

# Load the sample dataset
ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")


image = plt.imread(ds.attrs["frame_path"])

# Construct bins of size 30x30 pixels that cover the entire image
bin_pix = 30
bins = [
    np.arange(0, image.shape[0] + bin_pix, bin_pix),
    np.arange(0, image.shape[1] + bin_pix, bin_pix),
]

# Initialize the figure and axis
fig, ax = plt.subplots()

# Show the image
ax.imshow(image)

# Plot the occupancy 2D histogram for each individual
_, _, hist_data = plot_occupancy(
    da=ds.position,
    individuals=["individual1", "individual2"],
    keypoints="tailbase",
    ax=ax,
    alpha=0.5,
    bins=bins,
    cmin=3,  # Set the minimum shown count
    norm="log",
)

# Set the axis limits to match the image
ax.set_xlim(0, image.shape[1])
ax.set_ylim(image.shape[0], 0)

# %%
