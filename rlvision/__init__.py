"""Init rlvision."""

from __future__ import print_function

import os


HOME = os.environ["HOME"]
RLVISION_ROOT = os.path.join(HOME, ".rlvision")
RLVISION_DATA = os.path.join(RLVISION_ROOT, "data")
RLVISION_MODEL = os.path.join(RLVISION_ROOT, "model")

# setup the root
if not os.path.isdir(RLVISION_ROOT):
    os.makedirs(RLVISION_ROOT)
    print ("[MESSAGE] The rlvision root is at %s" % (RLVISION_ROOT))

if not os.path.isdir(RLVISION_DATA):
    os.makedirs(RLVISION_DATA)
    print ("[MESSAGE] The rlvision data directory is at %s" % (RLVISION_DATA))

if not os.path.isdir(RLVISION_MODEL):
    os.makedirs(RLVISION_MODEL)
    print ("[MESSAGE] The rlvision model directory is at %s"
           % (RLVISION_MODEL))
