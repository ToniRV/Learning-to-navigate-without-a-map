"""Vis vin data.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import os

import rlvision
from rlvision import utils
from rlvision.utils import process_map_data

#  import matplotlib.pyplot as plt

file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid16.pkl")

train, test = process_map_data(file_name)


for idx in xrange(100):
    #  pos = [train[1][idx]]
    pos = [(train[1][idx][1], train[1][idx][0])]
    grid = 1-train[0][idx, 0]
    utils.plot_grid(grid, (16, 16),
                    start=None, pos=pos, goal=None, title=None)
