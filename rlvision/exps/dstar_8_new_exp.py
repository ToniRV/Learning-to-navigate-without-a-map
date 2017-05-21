"""A D* experiment with 8x8 grid.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import cPickle as pickle
import numpy as np

import rlvision
from rlvision import utils
from rlvision.grid import GridSampler, Grid
from rlvision.utils import process_map_data
from rlvision.dstar import Dstar

# general parameters

n_samples = 100  # use limited data
n_steps = 16  # twice much as the step
save_model = True  # if true, all data will be saved for future use
enable_vis = True  # if true, real time visualization will be enable

# setup result folder

file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid8_with_idx.pkl")
im_data, state_data, label_data, sample_idx = process_map_data(
    file_name, return_full=True)
sampler = GridSampler(im_data, state_data, label_data, sample_idx, (8, 8))

print ("[MESSAGE] EXPERIMENT STARTED!")
for grid_idx in xrange(0, len(sample_idx), 7):
    # get a grid
    grid, state, label, goal = sampler.get_grid(grid_idx)

    # define step map
    grid = grid[0]
    step_map = np.ones((8, 8))
    pos = [state[0, 1], state[0, 0]]
    path = [(pos[0], pos[1])]
    print (path)

    planner = Dstar(path[0], (goal[1], goal[0]),
                    grid.flatten(), (8, 8))
    errors, next_move = planner.replan(next_move_only=False)
    solution = []
    for pos in next_move:
        solution.append((pos[1], pos[0]))
    print (solution)
    print (next_move)
    print (state)
    print (goal)
    utils.plot_grid(grid, (8, 8),
                    start=path[0],
                    pos=solution,
                    goal=(goal[0], goal[1]))

    #  for setp in xrange(n_steps):
    #      # masked image
    #      masked_img = utils.mask_grid(pos,
    #                                   grid, 3)
    #      # step image
    #      step_map = utils.accumulate_map(step_map, masked_img)
    #      change = np.where(np.logical_xor(
    #              planner.grid, step_map.flatten()))[0]
    #      block_list = np.unravel_index(change, planner.imsize)
    #      print (block_list)
    #      for idx in xrange(block_list[0].shape[0]):
    #          planner.add_obstacle(block_list[0][idx], block_list[1][idx])
    #
    #      errors, next_move = planner.replan()
    #      planner.reset_start_pos(next_move)
    #      if not errors and enable_vis:
    #          utils.plot_grid(step_map, (8, 8),
    #                          start=(path[0][1], path[0][0]),
    #                          pos=path[1:],
    #                          goal=goal)
    #
    #      # collect new action
    #      pos[0] = next_move[0]
    #      pos[1] = next_move[1]
    #      path.append((pos[1], pos[0]))
    #
    #      if pos[0] == goal[0] and pos[1] == goal[1]:
    #          print ("[MESSAGE] FOUND THE PATH %i" % (grid_idx+1))
    #          break

    planner.kill_subprocess()
