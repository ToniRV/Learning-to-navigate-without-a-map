"""Load HDF5 data test.

Author: Yuhuang Hu
Email : duguyue100
"""
from __future__ import print_function

import rlvision.utils as utils

# let's load split 3 of grid 16x16
db, imsize = utils.load_grid16(split=3)
print ("[MESSAGE] The data is loaded")

# the data keys are here
data_dict = utils.data_dict

print ("[MESSAGE] Data attributes")
print (data_dict)

# let's print the shape and the type of the grid data
print ("[MESSAGE]")
# call batch_im_data
batch_im_data = db["batch_im_data"]

print (batch_im_data.shape)
print (batch_im_data.dtype)

# The conclusion -- It's lighting fast!!!

# Let's plot a grid
utils.plot_grid(batch_im_data[1000, :], imsize)
