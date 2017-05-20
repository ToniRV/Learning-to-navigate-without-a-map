"""VIN PO Results.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle

import numpy as np

import rlvision

# input data
file_path = os.path.join(
    rlvision.RLVISION_MODEL, "grid8-po",
    "grid_8_result.pkl")

with open(file_path, "r") as f:
    data = pickle.load(f)
    f.close()

diff_arr = np.array(data['diff'], dtype=np.float32)

print (diff_arr)
print (diff_arr.mean())
