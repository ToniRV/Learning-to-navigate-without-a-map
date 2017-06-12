"""Export Value and Reward Map.

Get a model and export all the path and score
for grid 16x16

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
import keras.backend as K

import matplotlib.pyplot as plt

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
                         "chain_data", "grid16_with_idx.pkl")
model_file = os.path.join(
    rlvision.RLVISION_MODEL,
    "grid16-po", "vin-model-po-16-79-0.88.h5")

im_data, state_data, label_data, sample_idx = process_map_data(
    file_name, return_full=True)
model = vin_model(l_s=im_data.shape[2], k=20)
model.load_weights(model_file)

sampler = GridSampler(im_data, state_data, label_data, sample_idx, (16, 16))

gt_collector = []
po_collector = []
diff_collector = []


grid, state, label, goal = sampler.get_grid(14)
gt_collector.append(state)

step_map = np.zeros((2, 16, 16))
step_map[0] = np.ones((16, 16))
step_map[1] = grid[1]
pos = [state[0, 0], state[0, 1]]
path = [(pos[0], pos[1])]
start = (pos[0], pos[1])
for step in xrange(32):
    masked_img = utils.mask_grid((pos[1], pos[0]),
                                 grid[0], 3, one_is_free=False)
    step_map[0] = utils.accumulate_map(step_map[0], masked_img,
                                       one_is_free=False)

    action, reward, value = predict(step_map, pos, model, 20)
    dx, dy = get_action(action)
    pos[0] = pos[0] + dx
    pos[1] = pos[1] + dy
    path.append((pos[0], pos[1]))

    plt.figure()
    plt.imshow(value, cmap="jet")
    plt.colorbar()
    plt.scatter(x=[start[0]], y=[start[1]], marker="*", c="orange", s=50)
    plt.scatter(x=[pos[0]], y=[pos[1]], marker=".", c="purple", s=50)
    plt.scatter(x=[goal[0]], y=[goal[1]], marker="*", c="black", s=50)
    plt.savefig("grid16-14-%d.png" % (step), dpi=300)

    if pos[0] == goal[0] and pos[1] == goal[1]:
        print ("[MESSAGE] Found the path!")
        break
