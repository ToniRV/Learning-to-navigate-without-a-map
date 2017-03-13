"""Load data test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os

import rlvision
import rlvision.utils as utils

# file path

# for 8x8
#  file_name = "gridworld_8_3d_vision_50000.mat"
#  file_path = os.path.join(rlvision.RLVISION_DATA,
#                           "gridworld_8",
#                           file_name)

# for 16x16
file_name = "gridworld_40_3d_vision_"
file_path = os.path.join(rlvision.RLVISION_DATA,
                         "gridworld_40",
                         file_name)

save_dir = os.path.join(rlvision.RLVISION_DATA, "HDF5")

# save data
#  utils.create_grid_8_dataset(file_path, "gridworld_8.hdf5", save_dir)
#  utils.create_grid_16_dataset(file_path, "gridworld_16.hdf5", save_dir)
#  utils.create_grid_28_dataset(file_path, "gridworld_28.hdf5", save_dir)
utils.create_grid_40_dataset(file_path, "gridworld_40.hdf5", save_dir)
