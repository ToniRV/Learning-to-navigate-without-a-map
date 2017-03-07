"""Training VIN with tied weights.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np
import keras.backend as K
from keras.layers import Input
from keras.models import Model

from vin import model, utils

# load data
data_path = "./data/gridworld_8.mat"
imsize = 8

(Xtrain, S1train, S2train, ytrain,
 Xtest, S1test, S2test, ytest) = utils.process_gridworld_data(
         data_path,
         imsize=imsize)

if K.image_dim_ordering == "tf":
    Xtrain = np.transpose(Xtrain, (0, 2, 3, 1))
    Xtest = np.transpose(Xtest, (0, 2, 3, 1))
print ("[MESSAGE] Data is prepared.")

# build model
#  logits, nn = model.vi_block(X, S_V, S_H)

input_shape = Xtrain.shape[1:]
img_input = Input(shape=input_shape)

s_v_input = Input(shape=(S1train.shape[1:]))
s_h_input = Input(shape=(S2train.shape[1:]))

logits, nn = model.vi_block(img_input, s_v_input, s_h_input)

vin_model = Model([img_input, s_v_input, s_h_input], [logits, nn])
print ("[MESSAGE] The model is built")
