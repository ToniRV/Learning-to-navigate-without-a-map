"""DDPG implementation.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
from collections import deque
import random
import numpy as np

from keras.models import Model
from keras.layers import InputLayer, Input
from keras.layers import Dense
from keras.layers import add

import tensorflow as tf
import keras.backend as K


class ActorNet(object):
    """Actor network."""
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Init actor network.

        Parameters
        ----------
        state_size : tuple
            size of the state size
        """
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_size = action_size

        K.set_session(sess)

        # create model
        self.model, self.weights, self.state = \
            self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = \
            self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = \
            tf.gradients(self.model.output, self.weights,
                         -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = \
            tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        """Train network."""
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        """Target train."""
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = \
                self.tau*actor_weights[i]+(1-self.tau)*actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        """Create actor network."""
        print ("[MESSAGE] Build actor network.""")
        S = Input(shape=state_size)
        h_0 = Dense(300, activation="relu")(S)
        h_1 = Dense(600, activation="relu")(h_0)
        A = Dense(action_dim, activation="softmax")(h_1)

        model = Model(inputs=S, outputs=A)

        return model, model.trainable_weights, S


class CriticNet(object):
    """Critic Networks."""
    def __init__(self, sess, state_size, action_size,
                 batch_size, tau, learning_rate):
        """Init critic network."""
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_size = action_size

        K.set_session(sess)

        self.model, self.action, self.state = \
            self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = \
            self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        """Compute gradients for policy update."""
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        """Train target."""
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = \
                self.tau*critic_weights[i] + \
                (1-self.tau)*critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        """create critic network."""
        print ("[MESSAGE] Build critic network.""")
        S = Input(shape=state_size)
        A = Input(shape=(action_dim,))
        w_1 = Dense(300, activation="relu")(S)
        a_1 = Dense(600, activation="linear")(A)
        h_1 = Dense(600, activation="linear")(w_1)
        h_2 = add([h_1, a_1])
        h_3 = Dense(600, activation="relu")(h_2)
        V = Dense(action_dim, activation="softmax")(h_3)

        model = Model(inputs=[S, A], outputs=V)
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam")
        return model, A, S


class ReplayBuffer(object):
    """Reply Buffer."""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta*(mu-x)+sigma*np.random.randn(1)
