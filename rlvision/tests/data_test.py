"""Testing data."""

from __future__ import print_function
import os
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

import rlvision
from rlvision import utils

# prepare data
data_dir = "demo"
data_fn = "gridworld_8_3d_vision_demo.mat"
img_fn = "gridworld_8_3d_vision_demo.eps"
masked_img_fn = "gridworld_8_3d_vision_demo_masked.eps"
im_size = 8
radius = 2
idx = 4

#data_path = os.path.join(rlvision.RLVISION_DATA, data_dir, data_fn)
file_path = os.getcwd();
data_path = os.path.join(file_path, "../data/", data_dir, data_fn)

#save_path = os.path.join(rlvision.RLVISION_DATA, data_dir, img_fn)
#masked_save_path = os.path.join(rlvision.RLVISION_DATA, data_dir,
#                                masked_img_fn)

save_path = os.path.join(file_path, "../data/", data_dir, img_fn)
masked_save_path = os.path.join(file_path, "../data/", data_dir,
                                masked_img_fn)

Xdata, S1data, S2data, ydata, _, _, _, _, _, _, _, _ = \
    utils.process_gridworld_data(
        data_path, im_size)

target_x = S2data[idx]
target_y = S1data[idx]

plt.figure(figsize=(5, 5))
img_data = Xdata[0, 0, :, :]
img = np.asarray((1-Xdata[0, 0, :, :]), dtype=np.int8)*255
plt.imshow(img, cmap="gray")

goal = np.where(Xdata[0, 1, :, :] == np.max(Xdata[0, 1, :, :]))
goal = [goal[0][0], goal[1][0]]

plt.scatter(x=[goal[1]], y=[goal[0]], marker="*", c="r", s=50)
plt.scatter(x=S2data, y=S1data, marker=".", c="blue", s=50)

#  plt.axis("off")
plt.savefig(save_path, format="eps", dpi=100, bbox_inches="tight")

print ("[MESSAGE] The figure is saved at %s" % (save_path))

plt.figure(figsize=(5, 5))
mask = np.zeros((im_size, im_size))
rr, cc = draw.circle(target_y[0], target_x[0], radius=radius,
                     shape=mask.shape)
mask[rr, cc] = 1
masked_img = img*mask
plt.imshow(masked_img, cmap="gray")
plt.scatter(x=[goal[1]], y=[goal[0]], marker="*", c="r", s=50)
plt.scatter(x=target_x, y=target_y, marker=".", c="blue", s=50)

plt.savefig(masked_save_path, format="eps", dpi=100, bbox_inches="tight")
print ("[MESSAGE] The masked figure is saved at %s" % (masked_save_path))
