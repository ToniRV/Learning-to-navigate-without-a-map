"""Selection for PO grid EXPORT.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle
import numpy as np

import rlvision
from rlvision import utils
from rlvision.grid import GridSampler
from rlvision.utils import process_map_data
from rlvision.dstar import Dstar

# general parameters

n_samples = 100  # use limited data
n_steps = 32  # twice much as the step
save_model = True  # if true, all data will be saved for future use
enable_vis = True  # if true, real time visualization will be enable

# setup result folder

file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid28_with_idx.pkl")
im_data, state_data, label_data, sample_idx = process_map_data(
    file_name, return_full=True)
sampler = GridSampler(im_data, state_data, label_data, sample_idx, (28, 28))

gt_collector = []
po_collector = []
diff_collector = []

save_path = os.path.join(
    rlvision.RLVISION_MODEL,
    "grid28_paths")
if not os.path.isdir(save_path):
    os.makedirs(save_path)

print ("[MESSAGE] EXPERIMENT STARTED!")
for grid_idx in [77]:
    # get a grid
    grid, state, label, goal = sampler.get_grid(grid_idx)
    gt_collector.append(state)

    # define step map
    grid = 1-grid[0]
    step_map = np.ones((28, 28), dtype=np.uint8)
    pos = [state[0, 1], state[0, 0]]
    path = [(pos[0], pos[1])]

    planner = Dstar(path[0], (goal[1], goal[0]),
                    step_map.flatten(), (28, 28))

    for setp in xrange(n_steps):
        # masked image
        masked_img, coord = utils.mask_grid(pos,
                                            grid, 3, one_is_free=True)
        #  # step image
        step_map[coord[0], coord[1]] = grid[coord[0], coord[1]]
        #  step_map = utils.accumulate_map(step_map, masked_img)
        change = np.where(np.logical_xor(
                planner.grid, step_map.flatten()))[0]
        try:
            block_list = np.unravel_index(change, planner.imsize)
            print (block_list)
        except ValueError:
            break
        for idx in xrange(block_list[0].shape[0]):
            planner.add_obstacle(block_list[0][idx], block_list[1][idx])

        try:
            errors, next_move = planner.replan()
            planner.reset_start_pos(next_move)
        except ValueError:
            break
        if not errors and enable_vis:
            utils.plot_grid(step_map, (28, 28),
                            start=(path[0][1], path[0][0]),
                            pos=path[1:],
                            goal=(goal[0], goal[1]))
        # collect new action
        pos[0] = next_move[0]
        pos[1] = next_move[1]
        path.append((pos[1], pos[0]))

        if pos[0] == goal[1] and pos[1] == goal[0]:
            print ("[MESSAGE] FOUND THE PATH %i" % (grid_idx+1))
            break

    po_collector.append(path)
    diff_collector.append(abs(len(path)-1-state.shape[0]))
    print ("[MESSAGE] Diff %i" % (diff_collector[-1]))

    planner.kill_subprocess()

    data = {}
    data['environment'] = 1-grid
    data['gt'] = state
    path[0] = (path[0][1], path[0][0])
    print (path)
    data['po'] = path
    data['goal'] = goal
    with open(os.path.join(save_path, "grid_28_%i_dstar.pkl" % (grid_idx)),
              "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
