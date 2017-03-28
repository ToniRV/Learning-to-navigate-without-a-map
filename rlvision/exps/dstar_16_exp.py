"""A D* experiment with 16x16 grid.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import cPickle as pickle
import numpy as np

import rlvision
from rlvision import utils
from rlvision.grid import GridDataSampler, Grid
from rlvision.dstar import Dstar

# general parameters

n_samples = 100  # use limited data
n_steps = 32  # twice much as the step
save_model = True  # if true, all data will be saved for future use
enable_vis = True  # if true, real time visualization will be enable

# setup result folder

model_name = "dstar-16"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
print ("[MESSAGE] The model path is created at %s" % (model_path))

# load data
db, im_size = utils.load_grid16(split=1)

# prepare relevant data
im_data = db['im_data']
value_data = db['value_data']
states = db['state_xy_data']
label_data = db['label_data']
print ("[MESSAGE] The data is loaded.")

print ("[MESSAGE] Get data sampler...")
grid_sampler = GridDataSampler(im_data, value_data, im_size, states,
                               label_data)
print ("[MESSAGE] Data sampler ready.")

print ("[MESSAGE] EXPERIMENT STARTED!")
grid_id = 1
while grid_id <= n_samples and grid_sampler.grid_available:
    # sample grid
    print ("[MESSAGE] SAMPLING NEW GRID, Grid ID:", grid_id)
    grid, value, start_pos_list, pos_traj, goal_pos = grid_sampler.next()
    print ("[MESSAGE] New Grid is sampled.")
    print ("[MESSAGE] Number of trajectories:", len(start_pos_list))

    # carry out games
    print ("[MESSAGE] Carry out games...")
    result_pos_traj = []
    for start_pos in start_pos_list:
        # start a new game
        game = Grid(grid, value, im_size=im_size,
                    start_pos=start_pos, mask_radius=3,
                    dstar=True)
        planner = Dstar(game.start_pos, game.goal_pos,
                        game.dstar_curr_map.flatten(), game.im_size)
        # carry out game
        game_status = 0
        step = 1
        while True:
            print ("[MESSAGE] [IN GAME] Step:", step)
            errors, next_move = planner.replan()
            print (next_move)

            # TODO update game info
            # update game
            game.update_state(next_move)
            # update start position
            planner.reset_start_pos(next_move)
            # update grid
            change = np.where(np.logical_xor(
                planner.grid, game.dstar_curr_map.flatten()))[0]
            block_list = np.unravel_index(change, planner.imsize)
            print (block_list)
            for idx in xrange(block_list[0].shape[0]):
                planner.add_obstacle(block_list[0][idx], block_list[1][idx])

            # see if the game is ended
            _, game_status = game.get_state_reward()
            if game_status == 1:
                # success
                print ("[MESSAGE] The game is completed")
                break
            elif game_status == -1:
                print ("[MESSAGE] The game is failed")
                break

            if not errors:
                utils.plot_grid(game.curr_map, game.im_size,
                                start=game.start_pos,
                                pos=game.pos_history,
                                goal=game.goal_pos)
            else:
                print("[ERROR] Did not plot grid with path because of errors.")
            step += 1

        # save game
        result_pos_traj.append([game.pos_history, game_status])
    # save game result, save everything in a file
    if save_model:
        model_file = os.path.join(model_path, "dstar-16-%i.pkl" % (grid_id))
        with open(model_file, "wb") as f:
            pickle.dump([grid, value, im_size, start_pos_list,
                        pos_traj, goal_pos, result_pos_traj], f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        print ("[MESSAGE] The grid %i is saved at %s" % (grid_id, model_file))
    grid_id += 1

print ("[MESSAGE] EXPERIMENT FINISHED!")
