"""VIN training.

Grid 28x28

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import os
import cPickle as pickle

import keras.backend as K
from keras.callbacks import ModelCheckpoint

import rlvision
from rlvision import utils
from rlvision.vin import vin_model

# load data
file_name = os.path.join(rlvision.RLVISION_DATA,
                         "chain_data", "grid28_po.pkl")
model_path = os.path.join(rlvision.RLVISION_MODEL,
                          "grid28-po")
if not os.path.isdir(model_path):
    os.makedirs(model_path)

# parameters
batch_size = 256
nb_epochs = 80

print('# Minibatch-size: {}'.format(batch_size))
print('# epoch: {}'.format(nb_epochs))
print('')

train, test, _ = utils.process_map_data(file_name)
model = vin_model(l_s=train[0].shape[2], k=20)
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_file = os.path.join(
    model_path, "vin-model-po-28-{epoch:02d}-{acc:.2f}.h5")
checkpoint = ModelCheckpoint(model_file,
                             monitor='acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max',
                             save_weights_only=True)

history = model.fit([train[0].transpose((0, 2, 3, 1))
                     if K.image_data_format() == 'channels_last'
                     else train[0],
                     train[1]],
                    train[2],
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    callbacks=[checkpoint])

model_json = os.path.join(model_path, "vin_model_po_28.json")
with open(model_json, 'w') as f:
    f.write(model.to_json())

history_name = os.path.join(model_path, "history.pkl")
with open(history_name, "wb") as f:
    pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
print ("[MESSAGE] Save training history at %s" % (history_name))
