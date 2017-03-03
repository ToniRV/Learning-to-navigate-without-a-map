"""Define VIN model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import functools
import keras.backend as K
from keras import initializations
from keras.layers import Convolution2D


def vi_block(x, s_v, s_h, batch_size=10, n_iter=30,
             n_in=2, n_hid=150, n_q=10):
    """VI Block with tied weights.

    Parameters
    ----------
    x : tensor
        the input tensor
    s1 : tensor
        vertical position
    s2 : tensor
        horizontal position
    n_iter : int
        number of value iteration
    n_in : int
        number of input channels
    n_hid : int
        number of hidden channels
    n_q : int
        number of q layer (~action)
    """

    # the network
    # input = (batch, n_in,  h, w)
    # h     = (batch, n_hid, h, w)
    # r     = (batch, 1,     h, w)
    # q     = (batch, n_q  , h, w)

    # weight for q
    init = functools.partial(initializations.get("he_normal"),
                             dim_ordering=K.image_dim_ordering)
    if K.image_dim_ordering() == "tf":
        w_shape = (3, 3, 1, n_q)
    else:
        w_shape = (n_q, 1, 3, 3)
    w_q = init(w_shape)
    w_fb = init(w_shape)
    w_o = init((n_q, 8))

    h = Convolution2D(n_hid, 3, 3,
                      border_mode="same",
                      init="he_normal",
                      name="h0")(x)
    r = Convolution2D(1, 1, 1,
                      border_mode="same",
                      init="he_normal",
                      name="r", bias=False)(h)
    q = Convolution2D(n_q, 3, 3,
                      border_mode="same",
                      init="he_normal",
                      name="q", weights=w_q, bias=False)(r)
    # image channel axis
    if K.image_dim_ordering() == "tf":
        ch_axis = 3
        k_axis = 2
    else:
        ch_axis = 1
        k_axis = 1
    v = K.max(q, axis=ch_axis, keepdims=True)

    for i in range(0, n_iter-1):
        rv = K.concatenate([r, v], ch_axis)
        w_q_fb = K.concatenate([w_q, w_fb], k_axis)

        q = Convolution2D(n_q, 3, 3,
                          border_mode="same",
                          init="he_normal",
                          weights=w_q_fb, bias=False)(rv)
        v = K.max(q, axis=ch_axis, keepdims=True)

    q = Convolution2D(n_q, 3, 3,
                      border_mode="same",
                      init="he_normal",
                      weights=K.concatenate([w_q, w_fb], k_axis),
                      bias=False)(K.concatenate([r, v], ch_axis))

    # now using Theano indexing
    if K.image_dim_ordering == "tf":
        q = K.permute_dimensions(q, [0, 3, 2, 1])

    # select the convnet channels at the state position
    bs = K.shape(q)[0]
    rprn = K.reshape(K.repeat(K.reshape(K.arange(bs), [-1, 1]),
                     [1, batch_size]), [-1])
    ins_v = K.cast(K.reshape(s_v, [-1]), dtype="int32")
    ins_h = K.cast(K.reshape(s_h, [-1]), dtype="int32")

    idx_in = K.permute_dimensions(K.stack([ins_v, ins_h, rprn]), [1, 0])

    if K.image_dim_ordering == "tf":
        q = K.permute_dimensions(q, [2, 3, 0, 1])
    # using tf.gather_nd in the original code
    q_out = K.gather(q, idx_in)

    logits = K.dot(q_out, w_o)
    output = K.softmax(logits)

    return logits, output
