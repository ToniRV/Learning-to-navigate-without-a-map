"""Policy Gradient for Trained VIN 16x16.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import os
import numpy as np
import keras.backend as K

import rlvision
from rlvision import utils
from rlvision.vin import vin_model, Game, predict
from rlvision.utils import process_map_data
from rlvision.grid import GridSampler

gamma = 0.99
update_freq = 1
learning_rate = 0.001
num_output = 8
pg_model_file = "pg16_vin_model.h5"
pg_model_path = os.path.join(rlvision.RLVISION_MODEL, pg_model_file)


def discount_rewards(r):
    """Calculate discount rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# load data
file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid16_with_idx.pkl")
im_data, state_data, label_data, sample_idx = process_map_data(
    file_name, return_full=True)

# load model
model_file = os.path.join(rlvision.RLVISION_MODEL, "vin_model_po_16.h5")

k = 20

model = vin_model(l_s=im_data.shape[2], k=k)
model.load_weights(model_file)
model.compile(loss="categorical_crossentropy", optimizer="adam")

sampler = GridSampler(im_data, state_data, label_data, sample_idx, (16, 16))

reward_sum = 0
running_reward = None
episode_number = 0
xs_1, xs_2, dlogps, drs, probs = [], [], [], [], []
train_X, train_Y = [], []
num_victory = 0
for grid_idx in xrange(0, 700, 7):
    grid, state, label, goal = sampler.get_grid(grid_idx)
    game = Game(grid, state, label, goal)

    win_flag = False
    game_state = 0
    for step in xrange(32):
        # sample action
        aprob, action, _, _ = predict(game.step_map, game.pos, model, k)
        y = np.zeros((num_output,))
        y[action] = 1
        xs_1.append(game.step_map)
        xs_2.append(game.pos)
        probs.append(aprob.flatten())

        # update pos
        game.update_new_pos(action)
        if game.grid[0, game.pos[1], game.pos[0]] == 0.:
            # update map
            game.update_step_map(game.pos)

            game_state, reward = game.get_reward()

            dlogps.append(np.array(y).astype("float32"))
            reward_sum += reward
            drs.append(reward)
            if game_state == 1:
                # success
                win_flag = True
                break
        else:
            win_flag = False
            dlogps.append(np.array(y).astype("float32"))
            break

    if win_flag is False:
        game_state = -1
        reward_sum += -1.
        drs.append(-1.)

    if game_state in [1, -1]:
        episode_number += 1
        exp_1 = np.vstack(xs_1)
        exp_2 = np.vstack(xs_2)
        epdlogp = np.vstack(dlogps)
        print (epdlogp.shape)
        epr = np.vstack(drs)
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr
        # prepare training batch
        train_X.append(xs_1)
        train_Y.append(epdlogp)
        xs_1, xs_2, dlogps, drs = [], [], [], []

        if episode_number % update_freq == 0:
            y_train = probs + learning_rate*np.squeeze(
                train_Y)
            train_X = np.squeeze(np.vstack(train_X))
            if train_X.ndim < 4:
                train_X = np.expand_dims(train_X, axis=0)
            model.train_on_batch([train_X, exp_2],
                                 y_train)
            train_X, train_Y, probs = [], [], []
            os.remove(pg_model_path) \
                if os.path.exists(pg_model_path) else None
            model.save_weights(pg_model_path)
        running_reward = reward_sum if running_reward is None \
            else running_reward*0.99+reward_sum*0.01
        print ("Environment reset imminent. Total Episode "
               "Reward: %f. Running Mean: %f"
               % (reward_sum, running_reward))
        reward_sum = 0
        num_victory = num_victory+1 if game_state == 1 else num_victory
        print ("Episode %d Result: " % (episode_number) +
               ("Defeat!" if game_state == -1 else "Victory!"))
        print ("Successful rate: %d" %
               (num_victory))
