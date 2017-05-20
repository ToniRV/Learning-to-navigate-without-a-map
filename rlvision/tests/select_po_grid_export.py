"""Selection for PO grid EXPORT.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
import keras.backend as K

import rlvision
from rlvision import utils
from rlvision.vin import vin_model, get_layer_output
from rlvision.utils import process_map_data
from rlvision.grid import GridSampler


def get_action(a):
    if a == 0:
        return -1, -1
    if a == 1:
        return 0, -1
    if a == 2:
        return 1, -1
    if a == 3:
        return -1,  0
    if a == 4:
        return 1,  0
    if a == 5:
        return -1,  1
    if a == 6:
        return 0,  1
    if a == 7:
        return 1,  1
    return None


def find_goal(m):
    return np.argwhere(m.max() == m)[0][::-1]


def predict(im, pos, model, k):
    im_ary = np.array([im]).transpose((0, 2, 3, 1)) \
        if K.image_data_format() == 'channels_last' else np.array([im])
    res = model.predict([im_ary,
                         np.array([pos])])

    action = np.argmax(res)
    reward = get_layer_output(model, 'reward', im_ary)
    value = get_layer_output(model, 'value{}'.format(k), im_ary)
    reward = np.reshape(reward, im.shape[1:])
    value = np.reshape(value, im.shape[1:])

    return action, reward, value


file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid8_with_idx.pkl")
good_model_file = os.path.join(
    rlvision.RLVISION_MODEL, "grid8-po",
    "vin-model-po-8-79-0.99.h5")
bad_model_file = os.path.join(
    rlvision.RLVISION_MODEL, "grid8-po",
    "vin-model-po-8-00-0.69.h5")

im_data, state_data, label_data, sample_idx = process_map_data(
    file_name, return_full=True)

# good model
good_model = vin_model(l_s=im_data.shape[2], k=20)
good_model.load_weights(good_model_file)
print ("[MESSAGE] LOADED GOOD MODEL")

# bad model
bad_model = vin_model(l_s=im_data.shape[2], k=20)
bad_model.load_weights(bad_model_file)
print ("[MESSAGE] LOADED BAD MODEL")

sampler = GridSampler(im_data, state_data, label_data, sample_idx, (16, 16))

good_gt_collector = []
good_po_collector = []
good_diff_collector = []
bad_gt_collector = []
bad_po_collector = []
bad_diff_collector = []

save_path = os.path.join(
    rlvision.RLVISION_MODEL,
    "grid8_paths")
if not os.path.isdir(save_path):
    os.makedirs(save_path)

for grid_idx in [42, 77, 91, 105, 119, 126, 189]:
    # good model part
    grid, state, label, goal = sampler.get_grid(grid_idx)
    good_gt_collector.append(state)

    step_map = np.zeros((2, 8, 8))
    step_map[0] = np.ones((8, 8))
    step_map[1] = grid[1]
    pos = [state[0, 0], state[0, 1]]
    path = [(pos[0], pos[1])]
    for step in xrange(16):
        masked_img = utils.mask_grid((pos[1], pos[0]),
                                     grid[0], 3, one_is_free=False)
        step_map[0] = utils.accumulate_map(step_map[0], masked_img,
                                           one_is_free=False)

        action, _, _ = predict(step_map, pos, good_model, 20)
        dx, dy = get_action(action)
        pos[0] = pos[0] + dx
        pos[1] = pos[1] + dy
        path.append((pos[0], pos[1]))

        if pos[0] == goal[0] and pos[1] == goal[1]:
            print ("[MESSAGE] FOUND THE PATH %i" % (grid_idx+1))
            break

    good_po_collector.append(path)
    good_diff_collector.append(abs(len(path)-1-state.shape[0]))
    print ("[MESSAGE] Diff %i" % (good_diff_collector[-1]))

    data = {}
    data['environment'] = grid[0]
    data['gt'] = state
    data['po'] = path
    data['goal'] = goal
    with open(os.path.join(save_path, "grid_8_%i_good.pkl" % (grid_idx)),
              "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # bad model part
    bad_gt_collector.append(state)

    step_map = np.zeros((2, 8, 8))
    step_map[0] = np.ones((8, 8))
    step_map[1] = grid[1]
    pos = [state[0, 0], state[0, 1]]
    path = [(pos[0], pos[1])]
    for step in xrange(16):
        masked_img = utils.mask_grid((pos[1], pos[0]),
                                     grid[0], 3, one_is_free=False)
        step_map[0] = utils.accumulate_map(step_map[0], masked_img,
                                           one_is_free=False)

        action, _, _ = predict(step_map, pos, bad_model, 20)
        dx, dy = get_action(action)
        pos[0] = pos[0] + dx
        pos[1] = pos[1] + dy
        path.append((pos[0], pos[1]))

        if pos[0] == goal[0] and pos[1] == goal[1]:
            print ("[MESSAGE] FOUND THE PATH %i" % (grid_idx+1))
            break

    bad_po_collector.append(path)
    bad_diff_collector.append(abs(len(path)-1-state.shape[0]))
    print ("[MESSAGE] Diff %i" % (bad_diff_collector[-1]))

    data = {}
    data['environment'] = grid[0]
    data['gt'] = state
    data['po'] = path
    data['goal'] = goal
    with open(os.path.join(save_path, "grid_8_%i_bad.pkl" % (grid_idx)),
              "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    print ("[MESSAGE] Grid %i saved." % (grid_idx))

    # display good model and bad model jointly

#  data = {}
#  data['gt'] = gt_collector
#  data['po'] = po_collector
#  data['diff'] = diff_collector
#
#  with open("grid_28_result", "wb") as f:
#      pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#      f.close()
