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

db, imsize = utils.load_grid40(split=3)

# let's have a look at data

im_data = db['im_data']
value_data = db['value_data']
states = db['state_xy_data']
label_data = db['label_data']

print ("[MESSAGE] DATA LOADED")

grid_sampler = GridDataSampler(im_data, value_data, imsize, states,
                               label_data)

print ("[MESSAGE] SAMPLER READY")
grid, value, start_pos_list, pos_traj, goal_pos = grid_sampler.next()

print (grid)
print (value)
print (start_pos_list)
print (pos_traj)
print (goal_pos)

#  i = 0
#  while grid_sampler.grid_available:
#      print ("[MESSAGE] SAMPLING NEW GRID..")
#      grid, value, start_pos_list, pos_traj, goal_pos = grid_sampler.next()
#      i += 1
#      print ("[MESSAGE] THE %i-TH GRID SAMPLED. %i PATH FOUND." %
#             (i, len(start_pos_list)))

#  print (i)

#  states_xy = []
#  for i in xrange(100):
#      grid = np.reshape(im_data[i], imsize)
#      value = np.reshape(value_data[i], imsize)
#      goal_list = np.where(value == value.max())
#      # assume the first one
#      goal_pos = (goal_list[0][0], goal_list[1][0])
#      states_xy.append((states[i][0], states[i][1]))
#
#      utils.plot_grid(grid, imsize, states_xy, goal_pos, title=str(i))
