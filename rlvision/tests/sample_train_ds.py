"""Sample Train and test dataset.

Sample 6000 samples for training and testing.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import rlvision
from rlvision.grid import create_train_grid28

data_path = os.path.join(rlvision.RLVISION_DATA, "train", "gridworld_28")
if not os.path.isdir(data_path):
    os.makedirs(data_path)

create_train_grid28("gridworld_28", data_path, 250)
