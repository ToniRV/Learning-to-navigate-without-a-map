"""Selection for PO grid.

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


def plot_grid(data, imsize, start=None, pos=None, goal=None, title=None):
    """Plot a single grid with a vector representation.

    Parameters
    ----------
    data : numpy.ndarray
        the grid
    imsize : tuple
        the grid size
    pos : list
        list of tuple one want to draw
    goal : tuple
        the single goal
    """
    img = data.copy().reshape(imsize[0], imsize[1])
    img *= 255

    # good one
    plt.subplot(121)
    if start is not None:
        assert isinstance(start, tuple)
        plt.scatter(x=[start[0]], y=[start[1]], marker="*", c="orange", s=50)

    if pos is not None:
        assert isinstance(pos, list)
        for pos_element in pos[0][1:]:
            plt.scatter(x=[pos_element[0]], y=[pos_element[1]],
                        marker=".", c="blue", s=50)

    if goal is not None:
        assert isinstance(goal, tuple)
        plt.scatter(x=[goal[0]], y=[goal[1]], marker="*", c="r", s=50)

    plt.imshow(img, cmap="gray")
    if title is not None:
        assert isinstance(title, str)
        plt.title(title)

    # bad one
    plt.subplot(122)
    if start is not None:
        assert isinstance(start, tuple)
        plt.scatter(x=[start[0]], y=[start[1]], marker="*", c="orange", s=50)

    if pos is not None:
        assert isinstance(pos, list)
        for pos_element in pos[1][1:]:
            plt.scatter(x=[pos_element[0]], y=[pos_element[1]],
                        marker=".", c="blue", s=50)

    if goal is not None:
        assert isinstance(goal, tuple)
        plt.scatter(x=[goal[0]], y=[goal[1]], marker="*", c="r", s=50)

    plt.imshow(img, cmap="gray")
    if title is not None:
        assert isinstance(title, str)
        plt.title(title)

    plt.show()


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

sampler = GridSampler(im_data, state_data, label_data, sample_idx, (8, 8))

good_gt_collector = []
good_po_collector = []
good_diff_collector = []
bad_gt_collector = []
bad_po_collector = []
bad_diff_collector = []

for grid_idx in xrange(0, len(sample_idx), 7):
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

    plot_grid(1-grid[0], (8, 8),
              start=(state[0, 0], state[0, 1]),
              pos=[good_po_collector[-1], bad_po_collector[-1]],
              goal=(goal[0], goal[1]), title=None)

    # display good model and bad model jointly

#  data = {}
#  data['gt'] = gt_collector
#  data['po'] = po_collector
#  data['diff'] = diff_collector
#
#  with open("grid_28_result", "wb") as f:
#      pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#      f.close()
