"""Test train data loading.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from rlvision import grid

data, value, start_tot, traj_tot, goal_tot, imsize = grid.load_train_grid28()

print (data.shape)
print (value.shape)
print (len(start_tot))
print (len(traj_tot))
print (len(goal_tot))
print (imsize)
