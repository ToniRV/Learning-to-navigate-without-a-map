
import csv

import numpy as np
import scipy.io as sio

def process_gridworld_data(input, imsize):
  # run training from input matlab data file, and save test data prediction in output file
  # load data from Matlab file, including
  # im_data: flattened images
  # state_data: concatenated one-hot vectors for each state variable
  # state_xy_data: state variable (x,y position)
  # label_data: one-hot vector for action (state difference)
  im_size = [imsize, imsize]
  matlab_data = sio.loadmat(input)
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
  Xdata = np.transpose(Xdata,  (0, 2, 3, 1))
  S1data = state1_data.astype('int8')
  S2data = state2_data.astype('int8')

  return im_data

float_formatter = lambda x: "%.1d" % x
im_data = process_gridworld_data("../resources/gridworld_8.mat", 8)
i = 0
im_formatted = []
for line in im_data[1]
  if float_formatter(line) != "":
    im_formatted.append(float_formatter(line))
    i = i +1

import pdb; pdb.set_trace()  # breakpoint ec7f2b0e //
print(im_data)
with open('../resources/gridworld_8.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='', quoting=csv.QUOTE_NONE)
    writer.writerows(im_formatted)
