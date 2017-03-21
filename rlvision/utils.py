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
from skimage import draw
import matplotlib.pyplot as plt

import rlvision

# for the dataset labels
data_dict = ['batch_im_data', 'value_data', 'state_onehot_data',
             'state_xy_data', 'batch_value_data', 'batch_label_data',
             'label_data', 'state_y_data', 'im_data', 'state_x_data']


def mask_grid(pos, grid, radius, one_is_free=True):
    """Mask a grid.

    Parameters
    ----------
    pos : tuple
        the center of the circle (row, col)
        e.g. (2,3) = center at 3rd row, 4th column
    grid : numpy.ndarray
        should be a 2d matrix
        e.g. 8x8, 16x16, 28x28, 40x40
    imsize : tuple
        the grid size
    radius : int
        the length of the radius
    one_is_free : bool
        if True, then 1 is freezone, 0 is block
        if False, then 0 is freezone, 1 is block

    Returns
    -------
    masked_grid : numpy.ndarray
        the masked grid
    """
    mask = np.zeros_like(grid)
    rr, cc = draw.circle(pos[0], pos[1], radius=radius,
                         shape=mask.shape)
    mask[rr, cc] = 1
    if one_is_free:
        return grid*mask
    else:
        masked_img = np.ones_like(grid)
        masked_img[rr, cc] = grid[rr, cc]
        return masked_img


def accumulate_map(source_grid, new_grid, one_is_free=True):
    """Accumulate map.

    This function basically aggregate two grid.

    Parameters
    ----------
    source_grid : numpy.ndarray
        the source grid, assume 2d
    new_grid : numpy.ndarray
        the new grid to add on, assume 2d

    Returns
    -------
    acc_grid : numpy.ndarray
        the accumulated map
    """
    if one_is_free:
        acc_grid = source_grid+new_grid
        acc_grid[acc_grid > 0] = 1
        return acc_grid
    else:
        return source_grid*new_grid


def plot_grid(data, imsize, pos=None, goal=None, title=None):
    """Plot a single grid with a vector representation.

    Parameters
    ----------
    data : numpy.ndarray
        the grid
    imsize : tuple
        the grid size
    pos : list
        list of tuple one want to draw
    goal : tuple
        the single goal
    """
    img = data.copy().reshape(imsize[0], imsize[1])
    img *= 255

    if pos is not None:
        assert isinstance(pos, list)
        for pos_element in pos:
            plt.scatter(x=[pos_element[1]], y=[pos_element[0]],
                        marker=".", c="blue", s=50)

    if pos is not None:
        assert isinstance(goal, tuple)
        plt.scatter(x=[goal[1]], y=[goal[0]], marker="*", c="r", s=50)

    plt.imshow(data, cmap="gray")
    if title is not None:
        assert isinstance(title, str)
        plt.title(title)
    plt.show()


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


def add_h5_group(group_name, db, db_path=None):
    """Create a HDF5 group.

    Parameters
    ----------
    group_name : str
        the group name
    db : h5py.File
        the target database
    db_path : str
        assume to create group under
        db[db_path],
        create with root group if it's None
    """
    if db_path is None:
        gp = db
    else:
        if db_path not in db:
            print ("[MESSAGE] The path %s is not existed in this database" %
                   (db_path))
            gp = db
        else:
            gp = db[db_path]

    # add group if it's not existed
    if group_name not in gp:
        gp.create_group(group_name)
        print ("[MESSAGE] The group %s is created" % (group_name))

    db.flush()


def add_h5_ds(data, ds_name, db, group_name=None, data_type=np.uint8,
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
                      compression="gzip", dtype=dt)

    # flush the data
    db.flush()


def create_grid_8_dataset(mat_file_path, db_name, save_dir):
    """Convert grid 8x8 dataset from mat file to hdf5 format.

    Parameters
    ----------
    mat_file_path : str
        the path to the mat file
    db_name : str
        the name of the dataset
    save_dir : str
        the directory of the output path (must exist)
    """
    # load matlab data
    mat_data = load_mat_data(mat_file_path)
    print ("[MESSAGE] The Matlab data is loaded.")

    # init HDF5 database
    db = init_h5_db(db_name, save_dir)

    # save dataset
    for key in data_dict:
        add_h5_ds(mat_data[key], key, db)
        print ("[MESSAGE] Saved %s" % (key))

    db.flush()
    db.close()
    print ("[MESSAGE] The grid 8x8 dataset is saved at %s" % (save_dir))


def create_ds_from_splits(mat_file_prefix, db, splits):
    """Create datasets from splits of data.

    This function is created because of splits from larger datasets
    """
    for batch_idx in xrange(1, splits+1):
        # load data
        temp_data = load_mat_data(mat_file_prefix+str(batch_idx)+".mat")
        print ("[MESSAGE] Loaded %d-th split" % batch_idx)

        group_name = "grid_data_split_"+str(batch_idx)

        # save data within splits
        for key in data_dict:
            add_h5_group(group_name, db)
            add_h5_ds(temp_data[key], key, db, group_name)
            print ("[MESSAGE] %s: Saved %s" % (group_name, key))


def create_grid_16_dataset(mat_file_prefix, db_name, save_dir, splits=5):
    """Create 16x16 grid dataset in HDF5.

    Assume data file name as mat_file_prefix+str(index)+.mat

    Parameters
    ----------
    mat_file_prefix : str
        the prefix of the file name, such as
        /path/to/file/gridworld_16_3d_vision_
    db_name : str
        the name of the database
    save_dir : str
        the directory of the saved dataset
    splits : int
        number of splits for the dataset
    """
    # init HDF5 database
    db = init_h5_db(db_name, save_dir)

    # processing
    create_ds_from_splits(mat_file_prefix, db, splits)

    db.flush()
    db.close()
    print ("[MESSAGE] The grid 16x16 dataset is saved at %s" % (save_dir))


def create_grid_28_dataset(mat_file_prefix, db_name, save_dir, splits=20):
    """Create 28x28 grid dataset in HDF5.

    Assume data file name as mat_file_prefix+str(index)+.mat

    Parameters
    ----------
    mat_file_prefix : str
        the prefix of the file name, such as
        /path/to/file/gridworld_28_3d_vision_
    db_name : str
        the name of the database
    save_dir : str
        the directory of the saved dataset
    splits : int
        number of splits for the dataset
    """
    # init HDF5 database
    db = init_h5_db(db_name, save_dir)

    # processing
    create_ds_from_splits(mat_file_prefix, db, splits)

    db.flush()
    db.close()
    print ("[MESSAGE] The grid 28x28 dataset is saved at %s" % (save_dir))


def create_grid_40_dataset(mat_file_prefix, db_name, save_dir, splits=100):
    """Create 40x40 grid dataset in HDF5.

    Assume data file name as mat_file_prefix+str(index)+.mat

    Parameters
    ----------
    mat_file_prefix : str
        the prefix of the file name, such as
        /path/to/file/gridworld_40_3d_vision_
    db_name : str
        the name of the database
    save_dir : str
        the directory of the saved dataset
    splits : int
        number of splits for the dataset
    """
    # init HDF5 database
    db = init_h5_db(db_name, save_dir)

    # processing
    create_ds_from_splits(mat_file_prefix, db, splits)

    db.flush()
    db.close()
    print ("[MESSAGE] The grid 40x40 dataset is saved at %s" % (save_dir))


# default dataset loading function
# assume default location RLVISION_DATA/HDF5/data.hdf5
def load_grid8(return_imsize=True):
    """Load grid 8x8.

    Parameters
    ----------
    return_imsize : bool
        return a tuple with grid size if True

    Returns
    -------
    db : h5py.File
        a HDF5 file object
    imsize : tuple
        (optional) grid size
    """
    file_path = os.path.join(rlvision.RLVISION_DATA,
                             "HDF5", "gridworld_8.hdf5")
    if not os.path.isfile(file_path):
        raise ValueError("The dataset %s is not existed!" % (file_path))

    if return_imsize is True:
        return h5py.File(file_path, mode="r"), (8, 8)
    else:
        return h5py.File(file_path, mode="r")


def load_grid16(split=None, return_imsize=True):
    """Load grid 16x16.

    Parameters
    ----------
    split : int
        if not None and in the range of [1, 5]
        then return the specific split of data
    return_imsize : bool
        return a tuple with grid size if True

    Returns
    -------
    db : h5py.File
        a HDF5 file object
    imsize : tuple
        (optional) grid size
    """
    file_path = os.path.join(rlvision.RLVISION_DATA,
                             "HDF5", "gridworld_16.hdf5")
    if not os.path.isfile(file_path):
        raise ValueError("The dataset %s is not existed!" % (file_path))

    db = h5py.File(file_path, mode="r")
    if split is not None and split in xrange(1, 6):
        if return_imsize is True:
            return db["grid_data_split_"+str(split)], (16, 16)
        else:
            return db["grid_data_split_"+str(split)]
    else:
        if return_imsize is True:
            return db, (16, 16)
        else:
            return db


def load_grid28(split=None, return_imsize=True):
    """Load grid 28x28.

    Parameters
    ----------
    split : int
        if not None and in the range of [1, 20]
        then return the specific split of data
    return_imsize : bool
        return a tuple with grid size if True

    Returns
    -------
    db : h5py.File
        a HDF5 file object
    imsize : tuple
        (optional) grid size
    """
    file_path = os.path.join(rlvision.RLVISION_DATA,
                             "HDF5", "gridworld_28.hdf5")
    if not os.path.isfile(file_path):
        raise ValueError("The dataset %s is not existed!" % (file_path))

    db = h5py.File(file_path, mode="r")
    if split is not None and split in xrange(1, 21):
        if return_imsize is True:
            return db["grid_data_split_"+str(split)], (28, 28)
        else:
            return db["grid_data_split_"+str(split)]
    else:
        if return_imsize is True:
            return db, (28, 28)
        else:
            return db


def load_grid40(split=None, return_imsize=True):
    """Load grid 40x40.

    Parameters
    ----------
    split : int
        if not None and in the range of [1, 100]
        then return the specific split of data
    return_imsize : bool
        return a tuple with grid size if True

    Returns
    -------
    db : h5py.File
        a HDF5 file object
    imsize : tuple
        (optional) grid size
    """
    file_path = os.path.join(rlvision.RLVISION_DATA,
                             "HDF5", "gridworld_40.hdf5")
    if not os.path.isfile(file_path):
        raise ValueError("The dataset %s is not existed!" % (file_path))

    db = h5py.File(file_path, mode="r")
    if split is not None and split in xrange(1, 101):
        if return_imsize is True:
            return db["grid_data_split_"+str(split)], (40, 40)
        else:
            return db["grid_data_split_"+str(split)]
    else:
        if return_imsize is True:
            return db, (40, 40)
        else:
            return db
