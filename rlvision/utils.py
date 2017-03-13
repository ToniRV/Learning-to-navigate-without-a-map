"""Utility functions.

+ Data loading function
+ Data re-warping function
+ Drawing
"""
from __future__ import print_function
import os
import h5py
import numpy as np
import scipy.io as sio

import rlvision

# for the dataset labels
data_dict = ['batch_im_data', 'value_data', 'state_onehot_data',
             'state_xy_data', 'batch_value_data', 'batch_label_data',
             'label_data', 'state_y_data', 'im_data', 'state_x_data']


def process_gridworld_data(data_in, imsize):
    """Preprocess gridworld data from Matlab datafile.
    Note the output is in theano dimension
    (batch, height, weight, channels)

    Need restructure and investigation

    im_data: flattened images
    state_data: concatenated one-hot vectors for each state variable
    state_xy_data: state variable (x,y position)
    label_data: one-hot vector for action (state difference)
    """
    im_size = [imsize, imsize]
    matlab_data = sio.loadmat(data_in)
    im_data = matlab_data["batch_im_data"]
    im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
    value_data = matlab_data["batch_value_data"]
    state1_data = matlab_data["state_x_data"]
    state2_data = matlab_data["state_y_data"]
    label_data = matlab_data["batch_label_data"]
    ydata = label_data.astype('int8')
    Xim_data = im_data.astype('float32')
    Xim_data = Xim_data.reshape(-1, 1, im_size[0], im_size[1])
    Xval_data = value_data.astype('float32')
    Xval_data = Xval_data.reshape(-1, 1, im_size[0], im_size[1])
    Xdata = np.append(Xim_data, Xval_data, axis=1)
    # Need to transpose because Theano is NCHW, while TensorFlow is NHWC
    # use Theano dimension
    #  Xdata = np.transpose(Xdata,  (0, 2, 3, 1))
    S1data = state1_data.astype('int8')
    S2data = state2_data.astype('int8')

    all_training_samples = int(6/7.0*Xdata.shape[0])
    training_samples = all_training_samples
    Xtrain = Xdata[0:training_samples]
    S1train = S1data[0:training_samples]
    S2train = S2data[0:training_samples]
    ytrain = ydata[0:training_samples]

    Xtest = Xdata[all_training_samples:]
    S1test = S1data[all_training_samples:]
    S2test = S2data[all_training_samples:]
    ytest = ydata[all_training_samples:]
    ytest = ytest.flatten()

    sortinds = np.random.permutation(training_samples)
    Xtrain = Xtrain[sortinds]
    S1train = S1train[sortinds]
    S2train = S2train[sortinds]
    ytrain = ytrain[sortinds]
    ytrain = ytrain.flatten()
    return (Xdata, S1data, S2data, ydata, Xtrain,
            S1train, S2train, ytrain, Xtest, S1test, S2test, ytest)


def load_mat_data(file_path):
    """Load Matlab data and return all data objects.

    Parameters
    ----------
    file_path : str
        the file path of the dataset

    Returns
    -------
    mat_data : dict
        The dictionary with all the data.
    """
    if not os.path.isfile(file_path):
        raise ValueError("The file is not existed %s" % (file_path))

    return sio.loadmat(file_path)


def init_h5_db(db_name, save_dir):
    """Init a HDF5 database.

    Parameters
    ----------
    db_name : str
        a database name (with extension)
    save_dir : str
        a valid directory

    Returns
    -------
    database : h5py.File
        a HDF5 file object
    """
    # append extension if needed
    if db_name[-5:] != ".hdf5" and db_name[-3:] != ".h5":
        db_name += ".hdf5"

    # create destination folder if needed
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    db_name = os.path.join(save_dir, db_name)
    database = h5py.File(db_name, "a")

    return database


def add_h5_ds(data, ds_name, db, group_name=None, data_type=np.float32,
              renew=False):
    """Add a HDF5 dataset.

    Parameters
    ----------
    data : numpy.ndarry
        The data to be stored
    ds_name : str
        The name of the dataset
    db : h5py.File
        the database file object
    group_name : str
        Optional, the dataset will be placed under the
        group if it's vailable, otherwise will be
        with root group
    data_type : str
        Optional, Numpy supported data type
    renew : bool
        if True, the data will override the existing one.

    Returns
    -------
    None
    """
    dt = data_type
    if group_name is not None:
        if group_name not in db:
            group_name = None
            print ("[MESSAGE] the group %s is not existed, data will be save"
                   " at the root group" % (group_name))
            gp = db
        else:
            gp = db[group_name]
    else:
        gp = db

    if renew is True and ds_name in gp:
        del gp[ds_name]

    gp.create_dataset(ds_name, data=data.astype(dt),
                      data_type=dt)

    # flush the data
    db.flush()
