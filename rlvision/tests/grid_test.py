"""Test grid masking and accumulating.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import matplotlib.pyplot as plt

import rlvision.utils as utils

# load grid data
db, imsize = utils.load_grid40(split=30)
radius = 7
mask_pos_1 = (7, 7)
mask_pos_2 = (12, 12)

# plot a grid
grid_im = db[utils.data_dict[0]]

grid = 1-grid_im[0, :]

#  utils.plot_grid(grid, imsize)

grid = grid.reshape(imsize[0], imsize[1])
masked_grid_1 = utils.mask_grid(mask_pos_1, grid, radius, one_is_free=False)
masked_grid_2 = utils.mask_grid(mask_pos_2, grid, radius, one_is_free=False)
acc_grid = utils.accumulate_map(masked_grid_1, masked_grid_2,
                                one_is_free=False)

plt.figure(figsize=(10, 10))
plt.subplot(231)
plt.imshow((1-grid)*255, cmap="gray")
plt.axis("off")
plt.subplot(232)
plt.imshow((1-masked_grid_1)*255, cmap="gray")
plt.axis("off")
plt.subplot(233)
plt.imshow((1-masked_grid_2)*255, cmap="gray")
plt.axis("off")
plt.subplot(234)
plt.imshow((1-acc_grid)*255, cmap="gray")
plt.axis("off")
plt.show()
