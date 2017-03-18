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
grid_id = 1000
#utils.plot_grid(batch_im_data[grid_id, :], imsize)

# Let's run Dstar
import subprocess as sp
import os
import numpy as np
dir = os.getcwd()

grid= batch_im_data[grid_id, :]

# Call value_data
value_data = db["value_data"]

# Select a start position
start = [2, 5]

# Get start index
start_index = np.ravel_multi_index(start, imsize, order='F')
if grid[start_index] == 0:
    print("[ERROR] start position falls over an obstacle")
else:
    # Color in grey the start position
    grid[start_index] = 100

# Get goal index. It is given in the datasets, no need to verify that it is not
# an obstacle (I GUESS)
# Goal has always a value of 10 in the value_data? or just positive? or what?
goal_index = int(np.nonzero(value_data[grid_id, :] == 10)[0])

# Color goal cell in grey, but slighlty more grey than start
# TODO add colors to the grid instead
grid[goal_index] = 200

# Utility function.
stringify = lambda x: str(x).strip('[()]').replace(',', '').replace(' ', '').replace('\n', '')

print(batch_im_data[grid_id, :])
print(len(stringify(batch_im_data[grid_id, :])))

# Run dstar algorithm in c++
# Send start_index, goal_index, size of the grid and the grid through std input.
# All inputs must be flattened, aka string of int or ints (no matrices)
# I.e. a 2x2 grid would be given as a string of 4 ints (row-major, C style).
dstar_subprocess = sp.Popen([dir+"/dstar-lite/build/dstar_from_input",
                            stringify(start_index), stringify(goal_index), stringify(batch_im_data[grid_id, :])],
                            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

# Answer from dstar algorithm, it is send through stdout but also catches stderr.
response = dstar_subprocess.communicate()

# TODO Handle errors!

#batch_im_data[grid_id, 1]
# Response == (stdout answer, None)
answer = response[0].splitlines()
errors = response[1].splitlines()
print(answer)
print(errors)

# Let's plot Dstar ouput on the grid
utils.plot_grid(grid, imsize)


