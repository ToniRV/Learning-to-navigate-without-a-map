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

# Script Parameters
input_dim = imsize[0] * imsize[1]
update_frequency = 1
learning_rate = 0.001
collision_reward = -10
dropout_rate = 0
resume = False
render = False
data_format = "channels_first"
num_output = 8
model_file = "pg16_model.h5"
model_path = os.path.join(rlvision.RLVISION_MODEL, model_file)


# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
    # DQN Agent
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = input_dim
        # self.action_dim = env.action_space.n
        self.action_dim = num_output

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
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
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
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
          self.action_input:action_batch,
          self.state_input:state_batch
          })

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 32 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

#game information
game_idx = 1;

# initialize OpenAI Gym env and dqn agent
# env = gym.make(ENV_NAME)
# agent = DQN(env)
agent = DQN()

total_win = 0

for episode in xrange(EPISODE):
    # initialize task
    # state = env.reset()
    start_pos = start_tot[game_idx][3]
    print ("\nThis is game %d, start position %s" % (game_idx, map(str,start_pos)))
    game = grid.Grid(data[game_idx], value[game_idx], imsize,
                     start_pos, is_po=False)
    # Train
    for step in xrange(STEP):
        #update game state
        if step == 0:
            state = [[game.curr_map + 10 * value[game_idx].reshape(game.curr_map.shape) + 10 * game.curr_pos_map]]
            state = np.array(state).ravel()
        else:
            state = next_state
        #get next action from current state
        action = agent.egreedy_action(state) # e-greedy action for train
        game.update_state_from_action(action)
        reward, done = game.get_state_reward()
        next_state = [[game.curr_map + 10 * value[game_idx].reshape(game.curr_map.shape) + 10 * game.curr_pos_map]]
        next_state = np.array(next_state).ravel()
#         # Define reward for agent
#         if done == -1:
#             reward = -10
#         elif done == 1:
#             reward = 10
#         else:
#             reward = -0.1
        agent.perceive(state, action, reward, next_state, done)
        if done in [-1, 1]:
            if done == 1:
                total_win = total_win + 1.
            ## print exploration map
            map_explore = game.grid_map.astype(int)
            for i in range(len(game.pos_history)):
                map_explore[game.pos_history[i]] = i + 1
            map_explore[game.start_pos] = 33
            map_explore[game.goal_pos] = 99
            print(map_explore)
            print ("Episode %d Result: " % (episode) +
                   ("Victory!" if done == 1 else "Defeat!") +
                   (" Total steps: ") +
                   ('[%s]' % ', '.join(map(str, game.pos_history))))
            print("success rate: %f" % (total_win / (episode + 1.)))
            break

    # Test every 100 episodes
    if episode % 100 == 0:
        total_reward = 0
        for i in xrange(TEST):
            # state = env.reset()
            start_pos = start_tot[game_idx][3]
#             print ("\nThis is test %d, start position %s" % (i, map(str,start_pos)))
            game = grid.Grid(data[game_idx], value[game_idx], imsize,
                             start_pos, is_po=False)
            for j in xrange(STEP):
                #update game state
                if j == 0:
                    state = [[game.curr_map + 10 * value[game_idx].reshape(game.curr_map.shape) + 10 * game.curr_pos_map]]
                    state = np.array(state).ravel()
                else:
                    state = next_state
            #   action = agent.action(state) # direct action for test
            #   state,reward,done,_ = env.step(action)
                action = agent.egreedy_action(state) # e-greedy action for train
            #   next_state,reward,done,_ = env.step(action)
                game.update_state_from_action(action)
                reward, done = game.get_state_reward()
                next_state = [[game.curr_map + 10 * value[game_idx].reshape(game.curr_map.shape) + 10 * game.curr_pos_map]]
                next_state = np.array(next_state).ravel()

#                 if done == -1:
#                     reward = -10
#                 elif done == 1:
#                     reward = 10
#                 else:
#                     reward = -0.1

                total_reward += reward

                if done in [-1, 1]:
                    break

        ave_reward = total_reward/TEST
        print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward
        if ave_reward >= 9:
            break
