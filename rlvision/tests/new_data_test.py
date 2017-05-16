"""New data test."""
import os

import rlvision
from rlvision import utils

file_name = os.path.join(rlvision.RLVISION_DATA,
                         "new_data", "gridworld_16.mat")

db, im_size = utils.read_mat_data(file_name, 16)

train_data = utils.process_map_train_data(db, im_size)
