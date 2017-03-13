"""Load data test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os

import rlvision
import rlvision.utils as utils

# file path

file_name = "gridworld_16_3d_vision_1.mat"
file_path = os.path.join(rlvision.RLVISION_DATA,
                         "gridworld_16",
                         file_name)

data = utils.load_mat_data(file_path)

print (type(data))
print (data.keys())
print (data['__globals__'])
print (data['__header__'])
print (data['__version__'])
