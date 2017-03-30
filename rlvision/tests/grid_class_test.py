"""Grid class test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import numpy as np

from rlvision import utils
from rlvision.grid import Grid

# Load data

db, imsize = utils.load_grid40(split=3)
print ("[MESSAGE] The data is loaded")
print ("[MESSAGE] Image size: ", imsize)

# get a sample

sample_grid = db["batch_im_data"][0]
sample_value = db["batch_value_data"][0]

# set up grid

grid = Grid(sample_grid, sample_value, im_size=imsize)

# let's plot the grid first
utils.plot_grid(grid.grid_map, grid.im_size, grid.pos_history,
                grid.goal_pos)

# let's walk for some time

time = 30

for i in xrange(time):
    # draw action
    action = np.random.randint(8)
    # update grid
    grid.update_state_from_action(action)

    # plot grid
    utils.plot_grid(grid.curr_map, grid.im_size,
                    grid.pos_history, grid.goal_pos,
                    "the %d-th step" % (i))
    print ("[MESSAGE] reward: ", grid.get_state_reward())

print ("[MESSAGE] HISTORY")
print (grid.pos_history)
