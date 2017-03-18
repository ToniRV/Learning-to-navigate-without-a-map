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
dir = os.getcwd()

start = [0, 3]
goal = [3, 4]
grid= batch_im_data[grid_id, :]

stringify = lambda x: str(x).strip('[]').replace(',', '')

dstar_subprocess = sp.Popen([dir+"/dstar-lite/build/dstar_from_grid", stringify(start), stringify(goal), stringify(grid)],
 stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT)

response = dstar_subprocess.communicate()
#batch_im_data[grid_id, 1]
# Response == (stdout answer, None)
answer = response[0][0:3]

path = []
for s in answer.split(' '):
    if s.isdigit():
        path.append(int(s))

print(path)
index = (path[0]+1)*(path[1]+1)
print(index)

# Let's plot Dstar ouput on the grid
print(grid)
grid[index] = 100
utils.plot_grid(grid, imsize)


