"""Policy Gradient for Grid 16x16.

It's Keras 2!

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Activation

from rlvision import grid

# load data
data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid16()
print ("[MESSAGE] Data Loaded.")

# training 4000 samples, testing 1000 samples


# script parameters
input_dim = imsize[0]*imsize[1]
gamma = 0.99
update_freq = 1
learning_rate = 0.001
resume = False
network_type = "conv"
data_format = "channels_first"

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
    model.add(Dense(8, activation="softmax"))

# print model
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam")
print ("[MESSAGE] Model built.")

# training schedule
while True:
    # sample game

    # run and train game

    # update and save weight
