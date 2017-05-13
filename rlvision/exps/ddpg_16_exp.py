"""DDPG on 16x16 grid.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import numpy as np

import rlvision
from rlvision import grid


# load data
data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid16()
data = np.asarray(data, dtype="float32")
value = np.asarray(value, dtype="float32")
print ("[MESSAGE] Data Loaded.")
