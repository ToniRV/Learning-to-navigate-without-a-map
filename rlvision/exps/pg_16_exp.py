"""Policy Gradient for Grid 16x16.

It's Keras 2!

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Activation
from keras.regularizers import l2

import rlvision
from rlvision import grid


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

# define model
model = Sequential()
if network_type == "conv":
    model.add(Conv2D(32, (7, 7), padding="same",
                     input_shape=(2, imsize[0], imsize[1]),
                     kernel_regularizer=l2(0.0001),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (5, 5), padding="same",
                     kernel_regularizer=l2(0.0001),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (5, 5), padding="same",
                     kernel_regularizer=l2(0.0001),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (5, 5), padding="same",
                     kernel_regularizer=l2(0.0001),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(AveragePooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), padding="same",
                     kernel_regularizer=l2(0.0001),
                     data_format=data_format))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(num_output, activation="softmax"))
else:
    model.add(InputLayer(input_shape=(2, imsize[0], imsize[1])))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(300))
    model.add(Activation("relu"))
    model.add(Dense(300))
    model.add(Activation("relu"))
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
num_victory = 0
# go through entire game space
while True:
    for game_idx in xrange(num_train):
        for start_pos in start_tot[game_idx]:
            game = grid.Grid(data[game_idx], value[game_idx], imsize,
                             start_pos, is_po=False)
            # until the game is failed
            while True:
                #  game_state = game.get_state()
                #  plt.subplot(1, 3, 1)
                #  plt.imshow(game_state[0, 0], cmap="gray")
                #  plt.subplot(1, 3, 2)
                #  plt.imshow(game_state[0, 1], cmap="gray")
                #  plt.subplot(1, 3, 3)
                #  plt.imshow(game_state[0, 2], cmap="gray")
                #  plt.show()
                #  print (game_state[0, 0])
                # compute probability
                aprob = model.predict(game.get_state()).flatten()
                # sample feature
                xs.append(game.get_state())
                probs.append(model.predict(game.get_state()).flatten())
                # sample decision
                aprob = aprob/np.sum(aprob)
                action = np.random.choice(num_output, 1, p=aprob)[0]
                action_flag = game.is_pos_valid(game.action2pos(action))
                y = np.zeros((num_output,))
                if action_flag is True:
                    y[action] = 1
                    # update game and get feedback
                    game.update_state_from_action(action)
                    # if the game finished then train the model
                    reward, state = game.get_state_reward()
                # halt game if the action is hit the obstacle
                elif action_flag is False:
                    reward = -1.
                    state = -1
                dlogps.append(np.array(y).astype("float32")-aprob)
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
                    num_victory = num_victory+1 if state == 1 else num_victory
                    print ("Episode %d Result: " % (episode_number) +
                           ("Defeat!" if state == -1 else "Victory!"))
                    print ("Successful rate: %d" %
                           (num_victory))
                    # to next game
                    break
