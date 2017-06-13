"""VIN.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, merge, Reshape, Dense, Lambda, add
import keras.backend as K

from rlvision import utils

data_format = K.image_data_format()


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if data_format == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def vin_model(l_s=16, k=10, l_h=150, l_q=10, l_a=8):
    _handle_dim_ordering()

    def ext_start(inputs):
        m = inputs[0]
        s = inputs[1]
        w = K.one_hot(s[:, 0] + l_s * s[:, 1], l_s * l_s)
        return K.transpose(
            K.sum(w * K.permute_dimensions(m, (1, 0, 2)), axis=2))

    map_in = Input(shape=(l_s, l_s, 2)
                   if data_format == 'channels_last' else (2, l_s, l_s))
    x = Conv2D(l_h, (3, 3), strides=(1, 1),
               activation='relu',
               padding='same')(map_in)
    r = Conv2D(1, (1, 1), strides=(1, 1),
               padding='valid',
               use_bias=False, name='reward')(x)
    conv3 = Conv2D(l_q, (3, 3), strides=(1, 1),
                   padding='same',
                   use_bias=False)
    conv3b = Conv2D(l_q, (3, 3), strides=(1, 1),
                    padding='same',
                    use_bias=False)
    q_ini = conv3(r)
    q = q_ini
    for idx in range(k):
        v = Lambda(lambda x: K.max(x, axis=CHANNEL_AXIS, keepdims=True),
                   output_shape=(l_s, l_s, 1)
                   if data_format == 'channels_last' else (1, l_s, l_s),
                   name='value{}'.format(idx + 1))(q)
        q = add([q_ini, conv3b(v)])

    if data_format == "channels_last":
        q = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)),
                   output_shape=(l_q, l_s, l_s))(q)

    q = Reshape(target_shape=(l_q, l_s * l_s))(q)
    s_in = Input(shape=(2,), dtype='int32')
    q_out = merge([q, s_in], mode=ext_start, output_shape=(l_q,))
    out = Dense(l_a, activation='softmax', use_bias=False)(q_out)
    return Model(inputs=[map_in, s_in], outputs=out)


def get_layer_output(model, layer_name, x):
    return K.function([model.layers[0].input],
                      [model.get_layer(layer_name).output])([x])[0]


def get_action(a):
    if a == 0:
        return -1, -1
    if a == 1:
        return 0, -1
    if a == 2:
        return 1, -1
    if a == 3:
        return -1,  0
    if a == 4:
        return 1,  0
    if a == 5:
        return -1,  1
    if a == 6:
        return 0,  1
    if a == 7:
        return 1,  1
    return None


def find_goal(m):
    return np.argwhere(m.max() == m)[0][::-1]


def predict(im, pos, model, k):
    im_ary = np.array([im]).transpose((0, 2, 3, 1)) \
        if K.image_data_format() == 'channels_last' else np.array([im])
    res = model.predict([im_ary,
                         np.array([pos])])

    action = np.argmax(res)
    reward = get_layer_output(model, 'reward', im_ary)
    value = get_layer_output(model, 'value{}'.format(k), im_ary)
    reward = np.reshape(reward, im.shape[1:])
    value = np.reshape(value, im.shape[1:])

    return res, action, reward, value


class Game(object):
    """Define a game."""
    def __init__(self, grid, state, label, goal):
        self.grid = grid
        self.step_map = np.zeros((2, 16, 16))
        self.step_map[0] = np.ones((16, 16))
        self.step_map[1] = grid[1]

        self.state = state
        self.label = label
        self.goal = goal

        self.pos = [state[0, 0], state[0, 1]]
        self.path = [(self.pos[0], self.pos[1])]

    def update_step_map(self, pos):
        masked_img = utils.mask_grid((pos[1], pos[0]),
                                     self.grid[0], 3, one_is_free=False)
        self.step_map[0] = utils.accumulate_map(
            self.step_map[0], masked_img,
            one_is_free=False)

    def update_new_pos(self, action):
        dx, dy = get_action(action)
        self.pos[0] = self.pos[0] + dx
        self.pos[1] = self.pos[1] + dy
        self.path.append((self.pos[0], self.pos[1]))

    def get_reward(self):
        if self.pos[0] == self.goal[0] and self.pos[1] == self.goal[1]:
            return 1, 1.
        else:
            return 0, 0.
