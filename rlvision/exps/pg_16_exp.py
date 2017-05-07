"""Policy Gradient for Grid 16x16.

It's Keras 2!

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Activation

import rlvision
from rlvision import grid


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
data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid16()
data = np.asarray(data, dtype="float32")
value = np.asarray(value, dtype="float32")
print ("[MESSAGE] Data Loaded.")

# training 4000 samples, testing 1000 samples
num_train = 1
num_test = 1000

# script parameters
input_dim = imsize[0]*imsize[1]
gamma = 0.99
update_freq = 1
learning_rate = 0.001
resume = False
network_type = "conv"
data_format = "channels_first"
num_output = 8
model_file = "pg16_model.h5"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_file)

# define model
model = Sequential()
if network_type == "conv":
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=(3, imsize[0], imsize[1]),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=(3, imsize[0], imsize[1]),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(AveragePooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=(3, imsize[0], imsize[1]),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(num_output, activation="softmax"))

# print model
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam")
if resume is True:
    model.load_weights(model_path)
print ("[MESSAGE] Model built.")

# training schedule
reward_sum = 0
running_reward = None
episode_number = 0
xs, dlogps, drs, probs = [], [], [], []
train_X, train_Y = [], []
# go through entire game space
while True:
    for game_idx in xrange(num_test):
        for start_pos in start_tot[game_idx]:
            game = grid.Grid(data[game_idx], value[game_idx], imsize,
                             start_pos)
            # until the game is failed
            while True:
                # compute probability
                aprob = model.predict(game.get_state()).flatten()
                # sample feature
                xs.append(game.get_state())
                probs.append(aprob)
                # sample decision
                action = np.random.choice(num_output, 1, p=aprob)[0]
                y = np.zeros((num_output,))
                y[action] = 1
                action_flag = game.is_pos_valid(game.action2pos(action))
                # update game and get feedback
                game.update_state_from_action(action)
                # if the game finished then train the model
                dlogps.append(np.array(y).astype("float32")-aprob)
                reward, state = game.get_state_reward()
                if action_flag is False:
                    reward = -1.
                else:
                    reward = 1.
                reward_sum += reward
                drs.append(reward)
                if state in [1, -1]:
                    episode_number += 1
                    exp = np.vstack(xs)
                    epdlogp = np.vstack(dlogps)
                    epr = np.vstack(drs)
                    discounted_epr = discount_rewards(epr)
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)
                    epdlogp *= discounted_epr
                    # prepare training batch
                    train_X.append(xs)
                    train_Y.append(epdlogp)
                    xs, dlogps, drs = [], [], []

                    if episode_number % update_freq == 0:
                        y_train = probs + learning_rate*np.squeeze(
                            np.vstack(train_Y))
                        train_X = np.squeeze(np.vstack(train_X))
                        if train_X.ndim < 4:
                            train_X = np.expand_dims(train_X, axis=0)
                        model.train_on_batch(train_X,
                                             y_train)
                        train_X, train_Y, probs = [], [], []
                        os.remove(model_path) \
                            if os.path.exists(model_path) else None
                        model.save_weights(model_path)
                    running_reward = reward_sum if running_reward is None \
                        else running_reward*0.99+reward_sum*0.01
                    print ("Environment reset imminent. Total Episode "
                           "Reward: %f. Running Mean: %f"
                           % (reward_sum, running_reward))
                    reward_sum = 0
                    print ("Episode %d Result: " % (episode_number) +
                           ("Defeat!" if state == -1 else "Victory!"))
                    # to next game
                    break
