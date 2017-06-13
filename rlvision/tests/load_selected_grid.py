"""Load selected grid."""
import os

import numpy as np

import rlvision
from rlvision import utils

# load file
file_path = os.path.join(
        rlvision.RLVISION_MODEL, "grid28_paths",
        "grid_28_77_bad.pkl")

grid, gt_path, po_path, goal = utils.load_grid_selection(file_path)

gt = []
for idx in xrange(gt_path.shape[0]):
    gt.append((gt_path[idx, 0], gt_path[idx, 1]))

# plot grid
utils.plot_grid(1-grid, (28, 28),
                start=po_path[0],
                pos=gt[1:],
                goal=goal)

# plot po grid
step_map = np.ones((28, 28))
path = [po_path[0]]
for step in xrange(1, len(po_path)):
    masked_img = utils.mask_grid((path[-1][1], path[-1][0]),
                                 grid, 3, one_is_free=False)
    step_map = utils.accumulate_map(step_map, masked_img,
                                    one_is_free=False)
    path.append(po_path[step])
    utils.plot_grid(1-step_map, (28, 28),
                    start=path[0],
                    pos=path[1:],
                    goal=goal)
