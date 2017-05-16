"""Vis vin data.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import os

import numpy as np

import rlvision
from rlvision import utils
from rlvision.utils import process_map_data
from rlvision.grid import GridSampler

file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid28_with_idx.pkl")

train, test, sample_idx = process_map_data(file_name)
sampler = GridSampler(train[0], train[1], train[2], sample_idx, (16, 16))

#  grid, state, label = sampler.get_grid(7)
#  print (state.shape)
#
#  prev_grid = 1-grid[0]
#  source_grid = np.zeros_like(prev_grid)
#  for idx in xrange(state.shape[0]):
#      #  pos = [train[1][idx]]
#      pos = [(state[idx][1], state[idx][0])]
#      new_grid = 1-grid[0]
#
#      if np.array_equal(prev_grid, new_grid):
#          masked_grid = utils.mask_grid(pos[0], new_grid, 3)
#          source_grid = utils.accumulate_map(source_grid, masked_grid)
#      else:
#          prev_grid = grid
#          source_grid = np.zeros_like(grid)
#      utils.plot_grid(source_grid, (16, 16),
#                      start=None, pos=pos, goal=None, title=None)

print (len(sample_idx))
