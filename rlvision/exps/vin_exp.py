"""VIN training.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import numpy as np

import keras.backend as K
#  from rlvion.vin import vin_model
#  from utils import process_map_data

from rlvision import utils
from rlvision.vin import vin_model

# load data
db, im_size = utils.load_grid16(split=1)

# split data
im_data = db['im_data']
value_data = db['value_data']
state_data = db['state_xy_data']
label_data = db['label_data']

im_data = np.reshape(im_data, (im_data.shape[0], im_size[0], im_size[1]))
value_data = np.reshape(value_data,
                        (value_data.shape[0], im_size[0], im_size[1]))
label_data = np.array([np.eye(1, 8, l)[0] for l in label_data[:, 0]])


num = im_data.shape[0]
num_train = num - num / 5
im_train = np.concatenate((np.expand_dims(im_data[:num_train], 1),
                           np.expand_dims(value_data[:num_train], 1)),
                          axis=1).astype(dtype=np.float32)
state_train = state_data[:num_train]
label_train = label_data[:num_train]

im_test = np.concatenate((np.expand_dims(im_data[num_train:], 1),
                          np.expand_dims(value_data[num_train:], 1)),
                         axis=1).astype(dtype=np.float32)
state_test = state_data[num_train:]
label_test = label_data[num_train:]

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

with open('vin_model_structure.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('vin_model_weights.h5', overwrite=True)
