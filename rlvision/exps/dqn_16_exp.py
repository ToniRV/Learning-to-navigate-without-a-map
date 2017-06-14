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

#setup result folder
dim = 16
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

#Script Parameters
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

#Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
START_EPSILON = 0.2 # value of epsilon at the beginning
END_EPSILON = 0.01 # value of epsilon in the end
REPLAY_SIZE = 10000 # buffer size
BATCH_SIZE = 32 # minibatch size

class DQN():
    global debug

    # DQN Agent
    def __init__(self):
        # init some parameters
        self.replay_buffer = deque()

        self.time_step = 0

        self.epsilon = START_EPSILON

        self.state_dim = input_dim

        self.action_dim = num_output

        #initialize weights and biases of deep q net
        self.weights = {
            'w1': tf.Variable(tf.random_normal([3, 3, 2, 150])),
            'w2': tf.Variable(tf.random_normal([1, 1, 150, 1])),
            'w3': tf.Variable(tf.random_normal([3,3,1,10])),
            'out': tf.Variable(tf.random_normal([dim*dim*10, num_output]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([150])),
            'b2': tf.Variable(tf.random_normal([1])),
            'b3': tf.Variable(tf.random_normal([10])),
            'out': tf.Variable(tf.random_normal([num_output]))
        }
        self.state_input = tf.placeholder("float",[None, self.state_dim[0] * self.state_dim[1], 2])
        keep_prob = tf.placeholder(tf.float32) # dropout probability

        #create deep q network
        self.deep_q_network(self.state_input, self.weights, self.biases, keep_prob)
        self.training_rules()

        # Initialize session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # saver
        self.saver = tf.train.Saver()

    def save_model(self):
        save_path = self.saver.save(self.session, ''.join(model_path + str(game_idx) + ".ckpt"))
        print("Model saved in file: %s" % save_path)

    # define conv net for 2d
    def conv2d(self, x, W, b, strides = 1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def deep_q_network(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, [-1, self.state_dim[0], self.state_dim[1], 2])
        conv1 = self.conv2d(x, weights['w1'], biases['b1'])
        conv2 = self.conv2d(conv1, weights['w2'], biases['b2'])
        conv3 = self.conv2d(conv2, weights['w3'], biases['b3'])
        fc1 = tf.reshape(conv3, [-1, dim*dim*10])
        self.Q_value = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    def training_rules(self):
        self.action_input = tf.placeholder("float",[None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def store_episodes(self, state, action, reward, next_state, done):
        one_hot = np.zeros(self.action_dim)
        one_hot[action] = 1
        self.replay_buffer.append((state, one_hot, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # get random samples from replay memory and create minibatch
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [sample[0] for sample in minibatch]
        action_batch = [sample[1] for sample in minibatch]
        reward_batch = [sample[2] for sample in minibatch]
        next_state_batch = [sample[3] for sample in minibatch]

        # get y value
        y_value_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})

        for i in range(BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_value_batch.append(reward_batch[i])
            else :
                y_value_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # run optimizer
        self.optimizer.run(feed_dict={self.y_input:y_value_batch,
          self.state_input:state_batch,self.action_input:action_batch})

    def epsilon_greedy(self, state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (START_EPSILON - END_EPSILON)/10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:state})[0])

    def weight_variable(self, shape, name = None):
        return tf.Variable(tf.truncated_normal(shape), name = name)

    def bias_variable(self,shape, name = None):
        return tf.Variable(tf.constant(0.01, shape = shape), name = name)

# ---------------------------------------------------------
EPISODE = 10000 # Episode limitation
STEP = 2*dim + 1 # Step limitation in an episode
TEST = 10 # The number of experiment tests in every 100 episode

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
            print ("Start position is marked 33; Goal is marked 99; Other index indicate the latest step number; 1 is free space; 0 is obstacle.")
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
                # get next action from current state
                action = agent.epsilon_greedy(state) # epsilon-greedy for exploration
                game.update_state_from_action(action)
                reward, done = game.get_state_reward()
                # use updated partially-observable map as next state
                tmp_value_pos = value[game_idx].copy()
                tmp_value_pos.reshape(dim, dim)[game.curr_pos] = 1
                next_state = np.array([game.curr_map.ravel(), tmp_value_pos]).transpose()
                agent.store_episodes(state, action, reward, next_state, done)
                if done in [-1, 1]:
                    if done == 1:
                        total_win = total_win + 1.
                        curr_win = curr_win + 1.
                    ## print exploration map with index as latest step number
                    map_explore = game.curr_map.astype(int)
                    for i in range(len(game.pos_history)):
                        map_explore[game.pos_history[i]] = i + 1
                    map_explore[game.start_pos] = 33 #indicate start position
                    map_explore[game.goal_pos] = 99 #indicate goal position
                    print(map_explore)
                    print ("Episode %d Result: " % (episode) +
                           ("Victory!" if done == 1 else "Defeat!") +
                           (" Total steps: ") +
                           ('[%s]' % ', '.join(map(str, game.pos_history))))
                    print("current success rate: %f; total success rate: %f" % (curr_win / (episode + 1.), total_win / total_games))
                    break

            # Test every 100 episodes and stop early if possible
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
                        action = agent.epsilon_greedy(state) # epsilon-greedy action for exploration
                        game.update_state_from_action(action)
                        reward, done = game.get_state_reward()
                        # use updated partially-observable map as next state
                        tmp_value_pos = value[game_idx].copy()
                        tmp_value_pos.reshape(dim, dim)[game.curr_pos] = 1
                        next_state = np.array([game.curr_map.ravel(), tmp_value_pos]).transpose()
                        total_reward += reward

                        if done in [-1, 1]:
                            break

                ave_reward = total_reward/TEST
                print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward
                #stop as soon as the agent is good
                if ave_reward >= 9:
                    break
    agent.save_model()
