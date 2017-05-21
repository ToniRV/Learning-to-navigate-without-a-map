"""Plot Loss figure."""

import os
import cPickle as pickle

import rlvision

import matplotlib.pyplot as plt

# 8x8 path
grid8_file = os.path.join(
    rlvision.RLVISION_MODEL, "grid8-po",
    "history.pkl")
# 16x16 path
grid16_file = os.path.join(
    rlvision.RLVISION_MODEL, "grid16-po",
    "history.pkl")
# 28x28 path
grid28_file = os.path.join(
    rlvision.RLVISION_MODEL, "grid28-po",
    "history.pkl")

# load
with open(grid8_file) as f:
    grid8_history = pickle.load(f)
    f.close()
with open(grid16_file) as f:
    grid16_history = pickle.load(f)
    f.close()
with open(grid28_file) as f:
    grid28_history = pickle.load(f)
    f.close()

fig, ax1 = plt.subplots(figsize=(10, 8))

epochs = range(1, 81)

grid8_line, = ax1.plot(epochs, grid8_history['loss'], label="8x8 loss",
                       linewidth=3, color="#F44336")
grid16_line, = ax1.plot(epochs, grid16_history['loss'], label="16x16 loss",
                        linewidth=3, color="#2196F3")
grid28_line, = ax1.plot(epochs, grid28_history['loss'], label="28x28 loss",
                        linewidth=3, color="#D500F9")
ax1.set_xlabel("epochs", size=15)
ax1.set_ylabel("loss", size=15)
ax1.tick_params('y')
handles = [grid8_line, grid16_line, grid28_line]
ax1.grid(True, which="both", linestyle="--", color="#ECEFF1")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=len(handles), handles=handles, fontsize=15,
           mode="expand", borderaxespad=0.)

save_path = os.path.join(rlvision.RLVISION_MODEL, "loss-fig")
plt.savefig(save_path+".eps", dpi=300, format="eps")
plt.savefig(save_path+".png", dpi=300, format="png")
