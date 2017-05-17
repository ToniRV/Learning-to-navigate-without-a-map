"""A deep reinforcement learning experiment with 12x16 grid.
Author: Shu Liu
Email : liush@student.ethz.ch

Modified based on following:
<It's Keras 2!
Author: Yuhuang Hu
Email : duguyue100@gmail.com>
"""

from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Flatten, Dropout
# from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D, AveragePooling2D
import time
import datetime

import rlvision
from rlvision import utils
# from rlvision.grid import Grid

from rlvision import grid

start_time = time.time()
log_file = open("expriment-log.txt", "a")
log_file.write("experiment at " +
               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
log_file.close()

# setup result folder
model_name = "drl-16"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
print ("[MESSAGE] The model path is created at %s" % (model_path))

# load data
data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid16()
data = np.asarray(data, dtype="float32")
value = np.asarray(value, dtype="float32")
print ("[MESSAGE] Data Loaded.")

# training 4000 samples, testing 1000 samples
num_train = 4000
num_test = 1000 #not yet used

# # load data
# db, im_size = utils.load_grid16(split=1)

# prepare relevant data
# im_data = db['im_data']
# value_data = db['value_data']
# states = db['state_xy_data']
# label_data = db['label_data']
# print ("[MESSAGE] The data is loaded.")

# print ("[MESSAGE] Get data sampler...")
# grid_sampler = GridDataSampler(im_data, value_data, im_size, states,
#                                label_data)
# print ("[MESSAGE] Data sampler ready.")

# Script Parameters
input_dim = imsize[0] * imsize[1]
gamma = 0.99
update_frequency = 1
learning_rate = 0.001
collision_reward = -10
dropout_rate = 0.5
resume = False
render = False
# network_type = "conv"
data_format = "channels_first"
num_output = 8
model_file = "pg16_model.h5"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_file)

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
#         if r[t] != 0:
#             running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Define the main model (WIP)
number_of_inputs = 8;
def learning_model(input_dim = [4, imsize[0], imsize[1]], model_type=1):
    model = Sequential()
    if model_type == 0:
        model.add(Conv2D(32, (7, 7), padding="same",
                     input_shape=(4, imsize[0], imsize[1]),
                     data_format=data_format))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (5, 5), padding="same",
                         data_format=data_format))
        model.add(Dropout(dropout_rate))
        model.add(Activation("relu"))
        model.add(AveragePooling2D(2, 2))
        model.add(Conv2D(32, (3, 3), padding="same",
                         data_format=data_format))
        model.add(Dropout(dropout_rate))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer="he_uniform", activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(512, kernel_initializer="he_uniform", activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_output, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
    else:
        # model.add(Convolution2D(64, 9, 9, subsample=(1, 1),
        #           border_mode='same', activation='relu', init='he_uniform', input_shape = input_dim))
        model.add(Conv2D(64, (4, 4), kernel_initializer="he_uniform",
                         activation="relu", input_shape = (4, imsize[0], imsize[1]),
                         padding="same", data_format = data_format, strides=(1, 1)))

        model.add(Flatten())
        # model.add(Dense(256, activation='relu', init='he_uniform'))
        model.add(Dense(512, kernel_initializer="he_uniform", activation="sigmoid"))
        # model.add(Dense(256, activation='relu', init='he_uniform'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(512, kernel_initializer="he_uniform", activation="sigmoid"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(number_of_inputs, activation="sigmoid"))
        opt = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
    if resume is True:
        model.load_weights(model_path)
    return model

#initialize
xs, dlogps, drs, probs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
train_X = []
train_Y = []

win = 0
win_after_10k = 0

model = learning_model(model_type = 1)
model.summary()

# go through entire game space
for game_idx in xrange(num_train):
    for start_pos in start_tot[game_idx]:
        print ("\nThis is game %d, start position %s" % (game_idx, map(str,start_pos)))
        game = grid.Grid(data[game_idx], value[game_idx], imsize,
                         start_pos, is_po=False)

        #### avoid specail cases(straight line) that lead to overfitting
        if game.goal_pos[1] == start_pos[1]:
            print (game.goal_pos)
            print (start_pos)
            continue

        while True:
            # Get the current observable map

            x = [[game.curr_map, game.explored_area, value[game_idx].reshape(game.explored_area.shape), game.curr_pos_map]]
#             x = [[game.curr_map, game.curr_pos_map]]
            x = np.array(x)
            # print (x.shape)
            aprob = model.predict(x, batch_size=1).flatten()
            xs.append(x)
            probs.append(aprob)
            aprob = aprob/np.sum(aprob)
            print(aprob)
            # print ("bug2")
#             print (aprob)
#             if aprob[0] == aprob[0]:
#                 print(game.curr_map)
#                 print(game.explored_area)
#                 print(value[game_idx].reshape(game.explored_area.shape))
#                 print(game.curr_pos_map)

            # action = np.argmax(aprob)
            action = np.random.choice(num_output, 1, p=aprob)[0]
            action_flag = game.is_pos_valid(game.action2pos(action))

            y = np.zeros((num_output,))
            if action_flag is True:
                y[action] = 1
                game.update_state_from_action(action)
                reward, state = game.get_state_reward()
            # halt game if the action is hit the obstacle
            elif action_flag is False:
                #keep going after hitting the wall
                y = np.ones_like(y)
                y[action] = 0
                game.update_state_from_action(action)
                reward = collision_reward
                _ , state = game.get_state_reward()

                #terminate after hitting the wall
                # reward = collision_reward
                # state = -1

            # print ("bug3")
            dlogps.append(np.array(y).astype('float32') - aprob)
            reward = np.float64(reward)
            reward_sum += reward
            drs.append(reward)
            # print ("bug4")
            # save game
            # result_pos_traj.append([game.pos_history, game_status])

            if state in [1, -1]:
                if state == 1:
                    win += 1
                    if episode_number > 10000:
                        win_after_10k += 1
                # print ("bug5")
                episode_number += 1
                epx = np.vstack(xs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)
                discounted_epr = discount_rewards(epr)
#                 discounted_epr -= np.mean(discounted_epr)
#                 discounted_epr /= np.std(discounted_epr)
                epdlogp *= discounted_epr
                # print ("bug6")
                # Slowly prepare the training batch
                train_X.append(xs)
                train_Y.append(epdlogp)
                xs, dlogps, drs = [], [], []
                # Periodically update the model
                if episode_number % update_frequency == 0:
                    y_train = probs + learning_rate * np.squeeze(np.vstack(train_Y))
                    train_X = np.squeeze(np.vstack(train_X))
                    if train_X.ndim < 4:
                        train_X = np.expand_dims(train_X, axis=0)
                    # print ("bug7")
                    model.train_on_batch(train_X, y_train)
                    # Clear the batch
                    train_X, train_Y, probs = [], [], []
                    # print ("bug8")
                # Reset the current environment nad print the current results
                running_reward = reward_sum if running_reward is None \
                    else running_reward * 0.99 + reward_sum * 0.01
                print ('Environment reset imminent. Total Episode \
                    Reward: %f. Running Mean: %f' % (reward_sum, running_reward))
                reward_sum = 0
                print ("Episode %d Result: " % (episode_number) +
                       ("Defeat!" if state == -1 else "Victory!") +
                       (" Total steps: ") +
                       ('[%s]' % ', '.join(map(str, game.pos_history))))
                # to next game
                break

# Save a checkpoint of the model
os.remove(model_path) \
    if os.path.exists(model_path) else None
model.save_weights(model_path)

#write result to logs
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)

log_file = open("expriment-log.txt", "a")
log_file.write("Total wins: %d (success rate: %0.4f);  Total wins after 10k games: %d (success rate: %0.4f) \n" \
               % (win, float(win)/episode_number, win_after_10k, \
                  float(win_after_10k)/(episode_number - 10000)))
log_file.write("Total number of episodes is %d \n" % episode_number)
log_file.write("Parameters: gamma = %0.4f; learning_rate = %0.4f; update_frequency = %0.4f; Reward for hitting wall = %0.4f \n" % (gamma, learning_rate, update_frequency, collision_reward))
log_file.write("%d:%02d:%02d \n\n" % (h, m, s))
log_file.close()
