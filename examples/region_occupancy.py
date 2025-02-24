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
from matplotlib import pyplot as plt

from movement import sample_data
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
# Plot occupancy
# --------------
# A quick way to get an impression about the relative time spent in
# different regions of the maze is to use the
# :func:`movement.plots.plot_occupancy` function.
# By default, this function will the occupancy of the centroid
# of all available keypoints, for the first individual
# in the dataset (in this case, the only individual).


# Load the frame and plo
frame = plt.imread(ds.frame_path)

plot_occupancy(
    ds,
)
