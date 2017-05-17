"""VIN training.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import os

import keras.backend as K

import rlvision
from rlvision import utils
from rlvision.vin import vin_model

# load data
file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid16_po.pkl")

# parameters
batch_size = 256
nb_epochs = 50

print('# Minibatch-size: {}'.format(batch_size))
print('# epoch: {}'.format(nb_epochs))
print('')

train, test, _ = utils.process_map_data(file_name)
model = vin_model(l_s=train[0].shape[2], k=20)
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([train[0].transpose((0, 2, 3, 1))
           if K.image_dim_ordering() == 'tf' else train[0],
           train[1]],
          train[2],
          batch_size=batch_size,
          epochs=nb_epochs)

model_json = os.path.join(rlvision.RLVISION_MODEL, "vin_model_po_16.json")
model_h5 = os.path.join(rlvision.RLVISION_MODEL, "vin_model_po_16.h5")
with open(model_json, 'w') as f:
    f.write(model.to_json())
model.save_weights(model_h5, overwrite=True)
