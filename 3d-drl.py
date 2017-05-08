"""A deep reinforcement learning experiment with 16x16 grid.
Author: Shu Liu
Email : liush@student.ethz.ch
"""

from __future__ import print_function
import os
import cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
import time
import datetime

import rlvision
from rlvision import utils
from rlvision.grid import GridDataSampler, Grid

start_time = time.time()
log_file = open("expriment-log.txt", "a")
log_file.write("experiment at " +
               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
log_file.close()


# general parameters
n_samples = 1000  # use limited data
n_steps = 32  # twice much as the step
save_model = True  # if true, all data will be saved for future use
enable_vis = True  # if true, real time visualization will be enable

# setup result folder
model_name = "drl-16"
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

# Script Parameters
input_dim = im_size[0] * im_size[1]
gamma = 0.8
update_frequency = 1
learning_rate = 0.0005
resume = False
render = False

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Define the main model (WIP)
number_of_inputs = 8;
def learning_model(input_dim = [3, im_size[0], im_size[1]], model_type=1):
    model = Sequential()
    if model_type == 0:
        model.add(Reshape((3, im_size[0], im_size[1]), input_shape=(input_dim,)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = RMSprop(lr=learning_rate)
    else:
        model.add(Convolution2D(64, 9, 9, subsample=(1, 1),
                  border_mode='same', activation='relu', init='he_uniform', input_shape = input_dim))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', init='he_uniform'))
        model.add(Dense(256, activation='relu', init='he_uniform'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    if resume is True:
        model.load_weights('pong_model_checkpoint.h5')
    return model

#initialize
xs, dlogps, drs, probs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
train_X = []
train_y = []

win = 0
win_after_30k = 0

model = learning_model()

# Begin training
grid_id = 1
while grid_id <= n_samples and grid_sampler.grid_available:

    # if episode_number > 10000:
    #     break

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

        while True:
            #####################
            # Get the current observable map
            x = [[game.curr_map, game.explored_area, value.reshape(game.explored_area.shape)]]
            x = np.array(x)
#             x.reshape(1, 3, x.shape[1], x.shape[2])
            # Predict probabilities(regressed value) from the Keras model
#             aprob = model.predict(x.reshape([3, x.shape[1] * x.shape[2]]),
#                             batch_size=1).flatten()
            aprob = model.predict(x, batch_size=1).flatten()

            # aprob = aprob/np.sum(aprob)
            # Sample action
            # action = np.random.choice(number_of_inputs, 1, p=aprob)
            # Append features and labels for the episode-batch
#             xs.append(x)
            xs.append(x)
            probs.append(aprob)
            aprob = aprob/np.sum(aprob)

            action = np.random.choice(number_of_inputs, 1, p=aprob)[0]

            next_move = action

            #Get update parameter
            y = np.zeros([number_of_inputs])
            y[next_move] = 1
            dlogps.append(np.array(y).astype('float32') - aprob)

            # see if the game is ended
            #observation, reward, done, info = env.step(action)
            reward, game_status = game.get_state_reward()
            if game_status == 1:
                # success
                win += 1
                if episode_number > 30000:
                    win_after_30k += 1
                print ("[MESSAGE] The game is completed")
                print ("[MESSAGE] The path:", game.pos_history)
            elif game_status == -1:
                print ("[MESSAGE] The game is failed")
                print ("[MESSAGE] The path:", game.pos_history)

            done = True if game_status != 0 else False

            if not done:
                game.update_state_from_action(action)

            if done :
                break
            else:
                None

        # save game
        result_pos_traj.append([game.pos_history, game_status])

        reward = np.float64(reward)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            discounted_epr = discount_rewards(epr)
#             discounted_epr -= np.mean(discounted_epr)
#             discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr
            # Slowly prepare the training batch
            train_X.append(xs)
            train_y.append(epdlogp)
            xs, dlogps, drs = [], [], []
            # Periodically update the model
            if episode_number % update_frequency == 0:
                y_train = probs + learning_rate * np.squeeze(np.vstack(train_y))
#                 y_train = learning_rate * np.squeeze(np.vstack(train_y))
                print ('Training Snapshot:')
                print (y_train)
                model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
                # Clear the batch
                train_X = []
                train_y = []
                probs = []
                # Save a checkpoint of the model
                os.remove('drl_model_checkpoint.h5') \
                    if os.path.exists('drl_model_checkpoint.h5') else None
                model.save_weights('drl_model_checkpoint.h5')
            # Reset the current environment nad print the current results
            running_reward = reward_sum if running_reward is None \
                else running_reward * 0.99 + reward_sum * 0.01
            print ('Environment reset imminent. Total Episode \
                Reward: %f. Running Mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
        if reward != 0:
            print ('Episode %d Result: ' % episode_number, 'Defeat!' if game_status == -1 else 'VICTORY!')

total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)

log_file = open("expriment-log.txt", "a")
log_file.write("Total wins: %d (success rate: %0.4f);  Total wins after 30k games: %d (success rate: %0.4f) \n" \
               % (win, float(win)/episode_number, win_after_30k, \
                  float(win_after_30k)/(episode_number - 30000)))
log_file.write("Total number of episodes is %d \n" % episode_number)
log_file.write("Parameters: gamma = %0.4f; learning_rate = %0.4f; update_frequency = %0.4f \n" % (gamma, learning_rate, update_frequency))
log_file.write("%d:%02d:%02d \n\n" % (h, m, s))
log_file.close()
