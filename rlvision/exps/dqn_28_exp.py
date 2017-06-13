import gym
import tensorflow as tf
import os
import numpy as np
import random
from collections import deque

import time
import datetime

import rlvision
from rlvision import utils
from rlvision import grid

# setup result folder
dim = 28
model_name = "drl-16"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
print ("[MESSAGE] The model path is created at %s" % (model_path))

# load data
if dim == 8:
    data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid8()
elif dim == 16:
    data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid16()
elif dim == 28:
    data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid28()

data = np.asarray(data, dtype="float32")
value = np.asarray(value, dtype="float32")
print ("[MESSAGE] Data Loaded.")

# training 4000 samples, testing 1000 samples
num_train = 4000
num_test = 1000 #not yet used

# Script Parameters
input_dim = imsize
update_frequency = 1
learning_rate = 0.001
collision_reward = -10
dropout_rate = 0
resume = False
render = False
data_format = "channels_first"
num_output = 8
model_file = "dqn"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_file)

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
    global debug

    # DQN Agent
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0

        self.epsilon = INITIAL_EPSILON

        self.state_dim = input_dim

        self.action_dim = num_output

        #ConvNet Store layers weight & bias
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 2, 150])),
            'wc2': tf.Variable(tf.random_normal([1, 1, 150, 1])),
            'wc3': tf.Variable(tf.random_normal([3,3,1,10])),
            'out': tf.Variable(tf.random_normal([dim*dim*10, num_output]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([150])),
            'bc2': tf.Variable(tf.random_normal([1])),
            'bc3': tf.Variable(tf.random_normal([10])),
            'out': tf.Variable(tf.random_normal([num_output]))
        }
        self.state_input = tf.placeholder("float",[None, self.state_dim[0] * self.state_dim[1], 2])
        y = tf.placeholder(tf.float32, [None, num_output])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        #end of ConvNet

        # self.create_Q_network()
        self.create_conv_network(self.state_input, self.weights, self.biases, keep_prob)
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # saver
        self.saver = tf.train.Saver()

    def save_model(self):
        # saver = tf.train.Saver({"W1": self.W1, "W2": self.W2, "b1": self.b1, "b2": self.b2})
        save_path = self.saver.save(self.session, ''.join(model_path + str(game_idx) + ".ckpt"))
        print("Model saved in file: %s" % save_path)

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides = 1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

#     def maxpool2d(self, x, k = 2):
#         # MaxPool2D wrapper
#         return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                               padding='SAME')

    def create_conv_network(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, [-1, self.state_dim[0], self.state_dim[1], 2])
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        fc1 = tf.reshape(conv3, [-1, dim*dim*10])
        self.Q_value = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})

        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
          self.y_input:y_batch,
          self.state_input:state_batch,
          self.action_input:action_batch
          })

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:state})[0])

    def weight_variable(self,shape, name = None):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial, name = name)

    def bias_variable(self,shape, name = None):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial, name = name)

        # ---------------------------------------------------------
        # Hyper Parameters
        EPISODE = 10000 # Episode limitation
        STEP = 2*dim + 1 # Step limitation in an episode
        TEST = 10 # The number of experiment test every 100 episode


# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 10000 # Episode limitation
STEP = 2*dim + 1 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

agent = DQN()

total_win = 0.
total_games = 0.

for game_idx in xrange(num_train):
    for start_idx in range(len(start_tot[game_idx])):
        print(start_tot[game_idx])
        start_pos = start_tot[game_idx][start_idx]
        curr_win = 0
        start_pos_flag = True
        for episode in xrange(EPISODE):
            total_games += 1.
            print ("\nThis is game %d, start position %d, %s" % (game_idx + 1, start_idx + 1, map(str,start_pos)))
            game = grid.Grid(data[game_idx], value[game_idx], imsize, start_pos = start_pos, is_po=True)
            if start_pos_flag:
                if start_pos != game.pos_history[0]:
                    start_pos = game.pos_history[0]
                    print (game.pos_history[0])
                start_pos_flag = False
            # Train
            for step in xrange(STEP):
                #update game state
                if step == 0:
                    tmp_value_pos = value[game_idx].copy()
                    tmp_value_pos.reshape(dim, dim)[game.curr_pos] = 1
                    state = np.array([game.curr_map.ravel(), tmp_value_pos]).transpose()
                else:
                    state = next_state
                #get next action from current state
                action = agent.egreedy_action(state) # e-greedy action for train
                game.update_state_from_action(action)
                reward, done = game.get_state_reward()
                tmp_value_pos = value[game_idx].copy()
                tmp_value_pos.reshape(dim, dim)[game.curr_pos] = 1
                next_state = np.array([game.curr_map.ravel(), tmp_value_pos]).transpose()
                agent.perceive(state, action, reward, next_state, done)
                if done in [-1, 1]:
                    if done == 1:
                        total_win = total_win + 1.
                        curr_win = curr_win + 1.
                    ## print exploration map
                    map_explore = game.curr_map.astype(int)
                    for i in range(len(game.pos_history)):
                        map_explore[game.pos_history[i]] = i + 1
                    map_explore[game.start_pos] = 33
                    map_explore[game.goal_pos] = 99
                    print(map_explore)
                    print ("Episode %d Result: " % (episode) +
                           ("Victory!" if done == 1 else "Defeat!") +
                           (" Total steps: ") +
                           ('[%s]' % ', '.join(map(str, game.pos_history))))
                    print("current success rate: %f; total success rate: %f" % (curr_win / (episode + 1.), total_win / total_games))
                    break

            # Test every 100 episodes
            if episode % 100 == 0:
                total_reward = 0
                for i in xrange(TEST):
                    game = grid.Grid(data[game_idx], value[game_idx], imsize,
                                     start_pos, is_po=True)
                    for j in xrange(STEP):
                        if j == 0:
                            tmp_value_pos = value[game_idx].copy()
                            tmp_value_pos.reshape(dim, dim)[game.curr_pos] = 1
                            state = np.array([game.curr_map.ravel(), tmp_value_pos]).transpose()
                        else:
                            state = next_state
                        action = agent.egreedy_action(state) # e-greedy action for train
                        game.update_state_from_action(action)
                        reward, done = game.get_state_reward()
                        tmp_value_pos = value[game_idx].copy()
                        tmp_value_pos.reshape(dim, dim)[game.curr_pos] = 1
                        next_state = np.array([game.curr_map.ravel(), tmp_value_pos]).transpose()
                        total_reward += reward

                        if done in [-1, 1]:
                            break

                ave_reward = total_reward/TEST
                print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward
                if ave_reward >= 9:
                    break
    agent.save_model()
