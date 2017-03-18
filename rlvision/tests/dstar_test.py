"""Run dstar given hdf5 data

Author: Antoni Rosinol
Email : tonirosinol
"""
from __future__ import print_function
import rlvision.utils as utils
import subprocess as sp
import os
import numpy as np

# let's load split 3 of grid 16x16
db, imsize = utils.load_grid16(split=3)

# call batch_im_data
batch_im_data = db["batch_im_data"]

# Get current working directory
# TODO I don't know if there is a better way?
dir = os.getcwd()

# Let's take a random grid
grid_id = 1000;

# Get grid
grid= batch_im_data[grid_id, :]

# Get value data containing the reward values.
# The database puts a number 10 wherever the goal is (I think)
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

# Run dstar algorithm in c++
# Send start_index, goal_index, size of the grid and the grid through std input
# All inputs must be flattened, aka string of int or ints (no matrices)
# I.e. a 2x2 grid would be given as a string of 4 ints (row-major, C style).
dstar_subprocess = sp.Popen([dir+"/dstar-lite/build/dstar_from_input",
                            stringify(start_index), stringify(goal_index),
                            stringify(batch_im_data[grid_id, :])],
                            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

# Answer from dstar algorithm.
# It is send through stdout but also catches stderr.
response = dstar_subprocess.communicate()

# TODO Handle errors!

#batch_im_data[grid_id, 1]
# Response == (stdout answer, None)
answer = response[0].splitlines()
errors = response[1].splitlines()

if len(errors) == 0:
    for a in answer:
        grid[int(a)] = 150
else:
    print("[ERROR] Errors found.")
    print(errors)

# Let's plot Dstar ouput on the grid
utils.plot_grid(grid, imsize)


