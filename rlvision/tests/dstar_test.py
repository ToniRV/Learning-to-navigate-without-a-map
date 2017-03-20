"""Run dstar given hdf5 data

Author: Antoni Rosinol
Email : tonirosinol
"""
from __future__ import print_function
import rlvision.utils as utils
import rlvision.dstar as dstar

# let's load split 3 of grid 16x16
db, imsize = utils.load_grid16(split=3)

# call batch_im_data
batch_im_data = db["batch_im_data"]

# Let's take a random grid
grid_id = 100;

# Get grid
grid = batch_im_data[grid_id, :]

# Select a start position
start = [1, 1]

 # I think that a goal index is given in the datasets.
# Goal has always a value of 10 in the value_data? or just positive? or what?
# goal_index = int(np.nonzero(value_data[grid_id, :] == 10)[0])

# Get a random goal index.
goal = [14, 14]

# Make a dstar
dstar_instance = dstar.Dstar(start, goal, grid, imsize)

# Run dstar
grid_with_path = dstar_instance.grid
errors = dstar_instance.errors

# Let's plot dstar ouput on the grid
if errors is False:
    utils.plot_grid(grid_with_path, imsize)
else:
    print("[ERROR] Did not plot grid with path because of found errors.")


