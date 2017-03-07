"""Utility functions.

+ Data loading function
+ Data re-warping function
+ Drawing
"""
from __future__ import print_function
import numpy as np
import scipy.io as sio

import rlvision


def process_gridworld_data(data_in, imsize):
    """Preprocess gridworld data from Matlab datafile.
    Note the output is in theano dimension
    (batch, height, weight, channels)

    Need restructure and investigation

    im_data: flattened images
    state_data: concatenated one-hot vectors for each state variable
    state_xy_data: state variable (x,y position)
    label_data: one-hot vector for action (state difference)
    """
    im_size = [imsize, imsize]
    matlab_data = sio.loadmat(data_in)
    im_data = matlab_data["batch_im_data"]
    im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
    value_data = matlab_data["batch_value_data"]
    state1_data = matlab_data["state_x_data"]
    state2_data = matlab_data["state_y_data"]
    label_data = matlab_data["batch_label_data"]
    ydata = label_data.astype('int8')
    Xim_data = im_data.astype('float32')
    Xim_data = Xim_data.reshape(-1, 1, im_size[0], im_size[1])
    Xval_data = value_data.astype('float32')
    Xval_data = Xval_data.reshape(-1, 1, im_size[0], im_size[1])
    Xdata = np.append(Xim_data, Xval_data, axis=1)
    # Need to transpose because Theano is NCHW, while TensorFlow is NHWC
    # use Theano dimension
    #  Xdata = np.transpose(Xdata,  (0, 2, 3, 1))
    S1data = state1_data.astype('int8')
    S2data = state2_data.astype('int8')

    all_training_samples = int(6/7.0*Xdata.shape[0])
    training_samples = all_training_samples
    Xtrain = Xdata[0:training_samples]
    S1train = S1data[0:training_samples]
    S2train = S2data[0:training_samples]
    ytrain = ydata[0:training_samples]

    Xtest = Xdata[all_training_samples:]
    S1test = S1data[all_training_samples:]
    S2test = S2data[all_training_samples:]
    ytest = ydata[all_training_samples:]
    ytest = ytest.flatten()

    sortinds = np.random.permutation(training_samples)
    Xtrain = Xtrain[sortinds]
    S1train = S1train[sortinds]
    S2train = S2train[sortinds]
    ytrain = ytrain[sortinds]
    ytrain = ytrain.flatten()
    return (Xdata, S1data, S2data, ydata, Xtrain,
            S1train, S2train, ytrain, Xtest, S1test, S2test, ytest)
