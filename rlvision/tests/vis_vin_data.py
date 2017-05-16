"""Vis vin data.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import os
import cPickle as pickle

import numpy as np

import rlvision
from rlvision import utils
from rlvision.utils import process_map_data
from rlvision.grid import GridSampler

file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid8_po.pkl")

im_data, state_data, label_data, sample_idx = process_map_data(
    file_name, return_full=True)
sampler = GridSampler(im_data, state_data, label_data, sample_idx, (28, 28))

for grid_id in xrange(len(sample_idx)):
    grid, state, label = sampler.get_grid(grid_id)
    start_idx = sample_idx[grid_id-1] if grid_id != 0 else 0

    for state_idx in xrange(state.shape[0]):
        utils.plot_grid(1-im_data[start_idx+state_idx][0], (8, 8),
                        pos=[(state[state_idx, 1], state[state_idx, 0])])

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

#  new_img_data = np.zeros_like(im_data)
#  for grid_id in xrange(len(sample_idx)):
#      # sample grid
#      grid, state, label = sampler.get_grid(grid_id)
#
#      grid_map = grid[0]
#      value_map = grid[1]
#      acc_map = np.ones_like(grid_map)
#
#      grid_collect = np.zeros((state.shape[0], 2, 28, 28))
#
#      for move_id in xrange(state.shape[0]):
#          masked_img = utils.mask_grid((state[move_id, 1], state[move_id, 0]),
#                                       grid_map, 3, one_is_free=False)
#          acc_map = utils.accumulate_map(acc_map, masked_img,
#                                         one_is_free=False)
#          step_map = np.zeros((2, 28, 28))
#          step_map[0] = acc_map.copy()
#          step_map[1] = value_map.copy()
#          grid_collect[move_id] = step_map.copy()
#
#      start_idx = sample_idx[grid_id-1] if grid_id != 0 else 0
#      end_idx = sample_idx[grid_id]
#      new_img_data[start_idx:end_idx] = grid_collect
#      print ("[MESSAGE] %i-th grid." % grid_id)
#
#  data = {}
#  data['im'] = new_img_data[:, 0]
#  data['value'] = new_img_data[:, 1]
#  data['state'] = state_data
#  data['label'] = label_data
#  data['sample_idx'] = sample_idx
#  with open('grid28_po.pkl', mode='wb') as f:
#      pickle.dump(data, f)
