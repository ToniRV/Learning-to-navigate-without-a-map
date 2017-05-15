"""DDPG on 16x16 grid.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import keras.backend as K

import rlvision
from rlvision import grid
from rlvision.ddpg import ActorNet, CriticNet, ReplayBuffer

#  import matplotlib.pyplot as plt


# load data
data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid16()
data = np.asarray(data, dtype="float32")
value = np.asarray(value, dtype="float32")
print ("[MESSAGE] Data Loaded.")

# global parameter
buffer_size = 100000
batch_size = 32
gamma = 0.99
tau = 0.001
lra = 0.0001  # learning rate to actor
lrc = 0.001  # learning rate to critic

action_dim = 8
state_dim = (imsize[0], imsize[1], 2)

vision = False

explore = 10000.
episode_count = 2000
max_steps = 100000
reward = 0
done = False
step = 0
epsilon = 1
indicator = 0
train_indictor = 1  # if 1 then train, 0 then test

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

actor = ActorNet(sess, state_dim, action_dim, batch_size, tau, lra)
critic = CriticNet(sess, state_dim, action_dim, batch_size, tau, lrc)
buff = ReplayBuffer(buffer_size)

# TODO: load weights


# the game loop
for game_idx in xrange(episode_count):
    print("Episode : " + str(game_idx) +
          " Replay Buffer " + str(buff.count()))
    for start_pos in [start_tot[0][0]]:
        # start game
        game = grid.Grid(data[game_idx], value[game_idx], imsize,
                         start_pos, is_po=False)
        done = False

        s_t = game.get_state()
        s_t = s_t.transpose((0, 2, 3, 1))

        total_reward = 0.
        while True:
            #  plt.subplot(1, 3, 1)
            #  plt.imshow(s_t[0, :, :, 0], cmap="gray")
            #  plt.subplot(1, 3, 2)
            #  plt.imshow(s_t[0, :, :, 1], cmap="gray")
            #  plt.show()
            #  print (game_state[0, 0])
            loss = 0
            epsilon -= 1.0/explore

            # predict action
            a_t = actor.model.predict(s_t)
            aprob = a_t[0]/np.sum(a_t)
            action = np.random.choice(action_dim, 1, p=aprob)[0]
            action_flag = game.is_pos_valid(game.action2pos(action))
            act_vec = np.zeros((action_dim))

            if action_flag is True:
                act_vec[action] = 1.
                game.update_state_from_action(action)
                r_t, game_state = game.get_state_reward()
                s_t1 = game.get_state()
                s_t1 = s_t1.transpose((0, 2, 3, 1))
                done = False
                buff.add(s_t[0], act_vec, r_t, s_t1[0], done)
            else:
                r_t = -1.
                s_t1 = game.get_state().flatten()
                done = True

            # make sure the game is terminated properly
            if game_state in [-1, 1]:
                done = True

            # do batch update
            batch = buff.get_batch(batch_size)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict(
                [new_states, actor.target_model.predict(new_states)])

            for k in xrange(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma*target_q_values[k]

            if (train_indictor):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            #  print("Episode", game_idx, "Step", step,
            #        "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break
        #  if (train_indictor):
        #      print ("Save model")

        print("TOTAL REWARD @ " + str(game_idx) + "-th Episode  : Reward " +
              str(total_reward))
        print("Total Step: " + str(step))
        print("")
print ("Finish.")
