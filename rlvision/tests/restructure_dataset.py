"""Restructure data.

As the original data has shuffled the data randomly,
therefore it's difficult to reconstruct the exact path.
Here we manually select the necessary data from raw data.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os

import numpy as np

import rlvision
import rlvision.utils as utils
from rlvision.grid import GridDataSampler

# import data

db, imsize = utils.load_grid40(split=2)

# let's have a look at data

im_data = db['im_data']
value_data = db['value_data']
states = db['state_xy_data']
label_data = db['label_data']

grid_sampler = GridDataSampler(im_data, value_data, imsize, states,
                               label_data)

grid, value, start_pos_list, pos_traj = grid_sampler.next()
grid, value, start_pos_list, pos_traj = grid_sampler.next()

print (grid)
print (value)
print (start_pos_list)
print (pos_traj)

states_xy = []
for i in xrange(100):
    grid = np.reshape(im_data[i], imsize)
    value = np.reshape(value_data[i], imsize)
    goal_list = np.where(value == value.max())
    # assume the first one
    goal_pos = (goal_list[0][0], goal_list[1][0])
    states_xy.append((states[i][0], states[i][1]))

    utils.plot_grid(grid, imsize, states_xy, goal_pos, title=str(i))
