"""VIN training.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import os
import numpy as np

import keras.backend as K

import rlvision
from rlvision import utils
from rlvision.vin import vin_model

# load data
file_name = os.path.join(rlvision.RLVISION_DATA,
                         "new_data", "gridworld_16.mat")

db, im_size = utils.read_mat_data(file_name, 16)

im_train, state_train, label_train = utils.process_map_train_data(db, im_size)

# parameters
batch_size = 256
nb_epochs = 50


print('# Minibatch-size: {}'.format(batch_size))
print('# epoch: {}'.format(nb_epochs))
print('')

#  train, test = process_map_data(args.data)
model = vin_model(l_s=im_train.shape[2], k=20)
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([im_train.transpose((0, 2, 3, 1))
           if K.image_dim_ordering() == 'tf' else im_train,
           state_train],
          label_train,
          batch_size=batch_size,
          epochs=nb_epochs)

model_json = os.path.join(rlvision.RLVISION_MODEL, "vin_model.json")
model_h5 = os.path.join(rlvision.RLVISION_MODEL, "vin_model.h5")
with open(model_json, 'w') as f:
    f.write(model.to_json())
model.save_weights(model_h5, overwrite=True)
