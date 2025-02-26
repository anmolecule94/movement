"""Compute time spent in regions of interest
============================================

Define regions of interest and compute the time spent in each region.
"""

# %%
# Motivation
# ----------
# In this example we will work with a dataset of a mouse navigating
# an [elevated plus maze](https://en.wikipedia.org/wiki/Elevated_plus_maze),
# which consists of two open and two closed arms. Because of the
# general aversion of mice to open spaces, we expect mice to prefer the
# closed arms of the maze. Therefore, the proportion of time spent in
# open/closed arms is often used as a measure of anxiety-like behaviour
# in mice, i.e. the more time spent in the open arms, the less anxious the
# mouse is.

# %%
# Imports
# -------
import numpy as np
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import filter_by_confidence
from movement.plots import plot_occupancy

# %%
# Load data
# ---------
# The elevated plus maze dataset is provided as part of ``movement``'s
# sample data. We load the dataset and inspect its contents.

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
print(ds)
print("-----------------------------")
print(f"Individuals: {ds.individuals.values}")
print(f"Keypoints: {ds.keypoints.values}")

# %%
# Do some basic filtering
# -----------------------
# We will drop points with low confidence.

position = filter_by_confidence(ds.position, ds.confidence, threshold=0.95)

# %%
# Plot occupancy
# --------------
# A quick way to get an impression about the relative time spent in
# different regions of the maze is to use the
# :func:`movement.plots.plot_occupancy` function.
# By default, this function will the occupancy of the centroid
# of all available keypoints, for the first individual
# in the dataset (in this case, the only individual).


# Load the frame and plo
image = plt.imread(ds.frame_path)
height, width, channel = image.shape

# Construct bins that cover the entire image
bin_pix = 30  # pixels
bins = [
    np.arange(0, width + bin_pix, bin_pix),
    np.arange(0, height + bin_pix, bin_pix),
]


fig, ax = plt.subplots()
ax.imshow(image)  # Show the image

# Plot the occupancy 2D histogram for the centroid of all keypoints
fig, ax, hist_data = plot_occupancy(
    da=ds.position,
    ax=ax,
    alpha=0.8,
    bins=bins,
    cmin=10,  # Set the minimum shown count
    norm="log",
)

ax.set_title("Occupancy heatmap")
# Set the axis limits to match the image
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.collections[0].colorbar.set_label("# frames")
