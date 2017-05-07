"""Grid class to have a uniform grid functions.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import h5py
import cPickle as pickle

import numpy as np

import rlvision
from rlvision import utils


class Grid(object):
    """A centralized class for managing grid."""

    def __init__(self, grid_map, value_map, im_size=None,
                 start_pos=None, is_po=True, mask_radius=3,
                 grid_type="one-is-free", dstar=False):
        """Init function.

        Parameters
        ----------
        grid_map : numpy.ndarry
            the grid map with one and zero
        value_map : numpy.ndarray
            the map that encodes the value data
        im_size : tuple
            the size of the map, if it's not defined,
            the size will be acquired from the grid_map,
            in this case, the grid map can not be a
            1-D vector
        start_pos : tuple
            the start position (x, y) in a tuple
            if it's None, then will be randomly drawn from a poll of
            available points.
        is_po : bool
            if True, then the map is partially observable
        mask_radius : int
            the radius of the visible field
        grid_type : string
            "one-is-free"
            "zero-is-free"
        dstar : bool
            D* explores the states in a different way,
            if True, there will be a dual state map
        """
        if grid_type == "one-is-free":
            self.one_is_free = True
            self.empty_value = 1
            self.ob_value = 0  # obstacle value in map
        elif grid_type == "zero-is-free":
            self.one_is_free = False
            self.empty_value = 0
            self.ob_value = 1
        else:
            raise ValueError("The grid type should be either 'one-is-free'"
                             " or 'zero-is-free'")

        self.dstar = dstar
        if self.dstar:
            self.dstar_one_is_free = not self.one_is_free
            self.dstar_empty_value = 1-self.empty_value
            self.dstar_ob_value = 1-self.ob_value

        # set if it's partially observable
        self.is_po = is_po
        self.mask_radius = mask_radius

        # supply as a 1-D vector
        if grid_map.ndim == 1:
            if im_size is not None:
                # reshape only if the size matches
                if im_size[0]*im_size[1] == grid_map.shape[0]:
                    self.grid_map = np.reshape(grid_map, im_size)
                    self.im_size = im_size
                else:
                    raise ValueError("The number of elements in the grid"
                                     " doesn't match with the given"
                                     " image size")
            else:
                raise ValueError("It's impossible to recover the grid"
                                 " without giving image size")
        elif grid_map.ndim == 2:
            if im_size is not None:
                assert grid_map.shape == im_size
            else:
                self.im_size = grid_map.shape
        else:
            raise ValueError("The class doesn't support more than 2 dims!")

        # check value data
        if value_map.ndim == 1:
            if self.im_size[0]*self.im_size[1] == value_map.shape[0]:
                self.value_map = np.reshape(value_map, self.im_size)
            else:
                raise ValueError("The number of elements in the value map"
                                 " doesn't match with the given"
                                 " image size")
        elif value_map.ndim == 2:
            if value_map.shape == self.im_size:
                self.value_map = value_map
            else:
                raise ValueError("The size of the value map doesn't match"
                                 " with the grid data")

        else:
            raise ValueError("The class doesn't support more than 2 dims!")

        # set status of the grid

        # set start position
        if not self.is_pos_valid(start_pos):
            self.rand_start_pos()
        else:
            self.set_start_pos(start_pos)

        # set goal position, based on value map
        self.set_goal_pos()

    def is_pos_valid(self, pos):
        """Check if the position is valid.

        Parameters
        ----------
        pos : tuple
            a tuple (x, y) that represent a number

        Returns
        -------
        flag : bool
            true if valid (one empty position)
            false if invalid
        """
        if pos is None:
            pos = (0, 0)
        assert isinstance(pos, tuple)

        if self.grid_map[pos[0], pos[1]] == self.empty_value:
            return True
        else:
            return False

    def rand_start_pos(self):
        """Choose a start position randomly."""
        free_list = np.where(self.grid_map == self.empty_value)
        pos_idx = np.random.randint(free_list[0].shape[0])
        self.set_start_pos((free_list[0][pos_idx], free_list[1][pos_idx]))

    def set_start_pos(self, start_pos):
        """Set the start position.

        Parameters
        ----------
        start_pos : tuple
            potential start position (x, y)

        Returns
        -------
        The start position will be set if valid
        """
        if self.is_pos_valid(start_pos):
            self.start_pos = start_pos
            # clear all the history caches TODO
            self.set_curr_pos(start_pos)
            self.curr_map = self.get_curr_visible_map(self.start_pos)
            if self.dstar:
                self.dstar_curr_map = self.get_curr_dstar_visible_map(
                    self.start_pos)
            self.pos_history = [start_pos]
        else:
            print ("[MESSAGE] WARNING: The position is not valid, nothing"
                   " changes. (by set_start_pos)")

    def set_goal_pos(self):
        """Set goal position based on the value map."""
        goal_list = np.where(self.value_map == self.value_map.max())
        # assume the first one
        self.goal_pos = (goal_list[0][0], goal_list[1][0])

    def set_curr_pos(self, curr_pos):
        if self.is_pos_valid(curr_pos):
            self.curr_pos = curr_pos
        else:
            print ("[MESSAGE] WARNING: The position is not a vaild point"
                   " (by set_curr_pos)")

    def update_curr_map(self, map_update, dstar_map_update=None):
        """Update current map.

        Parameters
        ----------
        map_update : numpy.ndarray
            the update to the current map
        """
        self.curr_map = utils.accumulate_map(self.curr_map, map_update,
                                             one_is_free=self.one_is_free)
        if self.dstar:
            self.dstar_curr_map = utils.accumulate_map(
                self.dstar_curr_map, dstar_map_update,
                one_is_free=self.dstar_one_is_free)

    def get_curr_dstar_visible_map(self, pos):
        """Get current visible field by given a valid position.

        For D* algorithm

        Parameters
        ----------
        pos : tuple
            a valid position (x, y)

        Returns
        -------
        curr_vis_map : numpy.ndarray
            return a partially visible map by given position.
            Not if it's fully observable,
            this function will report the entire map
        """
        if self.is_pos_valid(pos):
            if self.is_po:
                return utils.mask_grid(pos, self.grid_map, self.mask_radius,
                                       one_is_free=self.dstar_one_is_free)
            else:
                return self.grid_map

    def get_curr_visible_map(self, pos):
        """Get current visible field by given a valid position.

        Parameters
        ----------
        pos : tuple
            a valid position (x, y)

        Returns
        -------
        curr_vis_map : numpy.ndarray
            return a partially visible map by given position.
            Not if it's fully observable,
            this function will report the entire map
        """
        if self.is_pos_valid(pos):
            if self.is_po:
                return utils.mask_grid(pos, self.grid_map, self.mask_radius,
                                       one_is_free=self.one_is_free)
            else:
                return self.grid_map

    def update_state(self, pos_update):
        """Update state by given position.

        This describe the transition between states.
        Assume we are at one state t, and by given a
        position, this function will update the
        state to t+1

        Parameters
        ----------
        pos_update : tuple
            a position (x, y)
        """
        if self.is_pos_valid(pos_update):
            # append to the history
            self.pos_history.append(pos_update)
            # update the current position
            self.set_curr_pos(pos_update)
            # update current map
            self.update_curr_map(self.get_curr_visible_map(pos_update),
                                 self.get_curr_dstar_visible_map(pos_update))
        else:
            print ("[MESSAGE] WARNING: The position is not valid, nothing"
                   " is updated (by update_state)")

    def update_state_from_action(self, action, verbose=0):
        """Update state from action space.

        0 1 2
        3   4
        5 6 7

        Update state from action space as above.

        Parameters
        ----------
        action : int
            sample from 0 - 7
        """
        pos_update = list(self.curr_pos)
        if action in [0, 1, 2]:
            pos_update[0] -= 1
        elif action in [5, 6, 7]:
            pos_update[0] += 1

        if action in [0, 3, 5]:
            pos_update[1] -= 1
        elif action in [2, 4, 7]:
            pos_update[1] += 1

        if verbose == 1:
            print ("[MESSAGE] Original pos: ", self.curr_pos)
            print ("[MESSAGE] Updated pos : ", pos_update)

        self.update_state(tuple(pos_update))

    def get_time(self):
        """Get the number of states."""
        return len(self.pos_history)

    def get_state_reward(self, max_num_steps=None):
        """Return reward for the state.

        Returns
        -------
        reward : int
            return reward for the state
        state : int
            1 : success
           -1 : fail
            0 : continue
        """
        if max_num_steps is not None:
            num_steps = max_num_steps
        else:
            num_steps = self.im_size[0]+self.im_size[1]
        recent_pos = self.pos_history[-1]
        if recent_pos == self.goal_pos and \
           self.get_time() <= num_steps+1:
            # success
            return self.value_map[recent_pos[0], recent_pos[1]], 1
        elif self.get_time() > num_steps+1:
            # failed
            return -self.value_map[self.goal_pos[0], self.goal_pos[1]], -1
        else:
            return self.value_map[recent_pos[0], recent_pos[1]], 0


class GridDataSampler(object):
    """A grid data sampler from the raw data."""

    def __init__(self, grid_data, value_data, im_size, states_xy,
                 label_data):
        """Init grid data sampler.

        Pararmeters
        -----------
        im_data : numpy.ndarray
            The image data
        value_data : numpy.ndarray
            the value data
        im_size : tuple
            the size of the map
        states_xy : numpy.ndarry
            the one that encode the paths (n_states, 2)
        """
        self.grid_data = grid_data
        self.value_data = value_data

        if im_size[0]*im_size[1] == grid_data.shape[1]:
            self.im_size = im_size

        self.states_xy = states_xy
        self.label_data = label_data
        self.curr_idx = 0
        self.grid_available = True

    def compare_pos(self, pos1, pos2):
        """Compare position."""
        if pos1[0] == pos2[0] and pos1[1] == pos2[1]:
            return True
        return False

    def get_next_state(self, pos, action):
        """Get next state according to action."""
        new_pos = [0, 0]
        if action in [5, 0, 4]:
            new_pos[0] = pos[0]-1
        elif action in [7, 1, 6]:
            new_pos[0] = pos[0]+1
        else:
            new_pos[0] = pos[0]

        if action in [5, 3, 7]:
            new_pos[1] = pos[1]-1
        elif action in [4, 2, 6]:
            new_pos[1] = pos[1]+1
        else:
            new_pos[1] = pos[1]
        return tuple(new_pos)

    def get_goal_pos(self, value_grid):
        """Get goal position."""
        value_map = np.reshape(value_grid.copy(), self.im_size)
        goal_list = np.where(value_map == value_map.max())
        # assume the first one
        return (goal_list[0][0], goal_list[1][0])

    def next(self):
        """Sample next sample."""
        if self.grid_available:
            grid = self.grid_data[self.curr_idx]
            value = self.value_data[self.curr_idx]
            goal_pos = self.get_goal_pos(value)

            # find end block idx
            curr_idx = self.curr_idx
            while curr_idx != self.grid_data.shape[0]:
                if np.array_equal(grid, self.grid_data[curr_idx]):
                    curr_idx += 1
                else:
                    break

            # parse grid traj
            start_pos_list = []
            pos_traj = []
            flag = True
            for idx in xrange(self.curr_idx, curr_idx):
                curr_pos = (self.states_xy[idx][0], self.states_xy[idx][1])
                next_pos = self.get_next_state(curr_pos, self.label_data[idx])

                if flag is True:
                    # when find a new start
                    # append the new start to the list
                    start_pos_list.append(curr_pos)
                    # construct a new pos traj
                    temp_pos_traj = []
                    # start searching
                    flag = False
                # evaluate if the next state is the goal position
                if self.compare_pos(goal_pos, next_pos):
                    # if yes, for next state, a new start begin
                    flag = True
                    # append the temp pos traj to collector
                    pos_traj.append(temp_pos_traj)

                temp_pos_traj.append(curr_pos)

            # update current idx
            self.curr_idx = curr_idx
            if self.curr_idx == self.grid_data.shape[0]:
                self.grid_available = False

            return grid, value, start_pos_list, pos_traj, goal_pos
        else:
            print ("[MESSAGE] No grid available""")


def sample_data(db, imsize, num_samples=0):
    """Sample data from a database.

    Parameters
    ----------
    db : h5py.File
        a HDF 5 file object
    num_samples : int
        the number of samples

    Returns
    -------
    grid_data : numpy.ndarray
        the grid data
    value_data : numpy.ndarray
        the value data
    start_pos_list : list
        the list of start position
    pos_traj : list
        the list of position trajectory
    goal_pos : list
        the goal of position
    """
    # load data
    im_data = db['im_data']
    value_data = db['value_data']
    states = db['state_xy_data']
    label_data = db['label_data']

    # created a sampler
    grid_sampler = GridDataSampler(im_data, value_data, imsize,
                                   states, label_data)
    print ("[MESSAGE] Create a sampler")

    data_collector = []
    value_collector = []
    start_pos_collector = []
    pos_traj_collector = []
    goal_pos_collector = []

    idx = 0
    while grid_sampler.grid_available and idx < num_samples:
        grid, value, start_pos_list, pos_traj, goal_pos = grid_sampler.next()
        if len(start_pos_list) < 8:
            print ("[MESSAGE] THE %i-TH GRID SAMPLED. %i PATH FOUND." %
                   (idx, len(start_pos_list)))
            data_collector.append(grid)
            value_collector.append(value)
            start_pos_collector.append(start_pos_list)
            pos_traj_collector.append(pos_traj)
            goal_pos_collector.append(goal_pos)
            idx += 1

    data_collector = np.asarray(data_collector, dtype=np.uint8)
    value_collector = np.asarray(value_collector, dtype=np.uint8)

    if idx < num_samples:
        print ("[MESSAGE] %i samples collected." % (idx+1))
    return (data_collector, value_collector, start_pos_collector,
            pos_traj_collector, goal_pos_collector)


def sample_data_grid8(num_samples=0):
    """Sample data from 8x8 grid.

    Parameters
    ----------
    num_samples : int
        number of samples

    Return
    ------
    grid_data : numpy.ndarray
        the grid data
    value_data : numpy.ndarray
        the value data
    start_pos_list : list
        the list of start position
    pos_traj : list
        the list of position trajectory
    goal_pos : list
        the goal of position
    """
    db, imsize = utils.load_grid8()

    return sample_data(db, imsize, num_samples)


def sample_data_grid16(split=None, num_samples=0):
    """Sample data from 16x16 grid.

    Parameters
    ----------
    num_samples : int
        number of samples

    Return
    ------
    grid_data : numpy.ndarray
        the grid data
    value_data : numpy.ndarray
        the value data
    start_pos_list : list
        the list of start position
    pos_traj : list
        the list of position trajectory
    goal_pos : list
        the goal of position
    """
    db, imsize = utils.load_grid16(split)

    return sample_data(db, imsize, num_samples)


def sample_data_grid28(split=None, num_samples=0):
    """Sample data from 28x28 grid.

    Parameters
    ----------
    num_samples : int
        number of samples

    Return
    ------
    grid_data : numpy.ndarray
        the grid data
    value_data : numpy.ndarray
        the value data
    start_pos_list : list
        the list of start position
    pos_traj : list
        the list of position trajectory
    goal_pos : list
        the goal of position
    """
    db, imsize = utils.load_grid28(split)

    return sample_data(db, imsize, num_samples)


def sample_data_grid40(split=None, num_samples=0):
    """Sample data from 40x40 grid.

    Parameters
    ----------
    num_samples : int
        number of samples

    Return
    ------
    grid_data : numpy.ndarray
        the grid data
    value_data : numpy.ndarray
        the value data
    start_pos_list : list
        the list of start position
    pos_traj : list
        the list of position trajectory
    goal_pos : list
        the goal of position
    """
    db, imsize = utils.load_grid40(split)

    return sample_data(db, imsize, num_samples)


def create_train_grid8(db_name, save_dir, num_samples=0):
    """Create training dataset for 8x8 grid.

    Parameters
    ----------
    db_name : str
        the name of dataset
    save_dir : str
        the directory of the output path (must exist)
    """
    db = utils.init_h5_db(db_name+".h5", save_dir)

    # collect data
    (data_collector, value_collector, start_pos_collector,
     pos_traj_collector, goal_pos_collector) = sample_data_grid8(num_samples)

    # save data
    utils.add_h5_ds(data_collector, "data", db)
    utils.add_h5_ds(value_collector, "value", db)
    db.flush()
    db.close()

    with open(os.path.join(save_dir, db_name+"_start.pkl"), "w") as f:
        pickle.dump(start_pos_collector, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    with open(os.path.join(save_dir, db_name+"_traj.pkl"), "w") as f:
        pickle.dump(pos_traj_collector, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    with open(os.path.join(save_dir, db_name+"_goal.pkl"), "w") as f:
        pickle.dump(goal_pos_collector, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    print ("[MESSAGE] Save dataset at %s" % (save_dir))


def create_train_grid16(db_name, save_dir, num_samples=0):
    """Create training dataset for 16x16 grid.

    Parameters
    ----------
    db_name : str
        the name of dataset
    save_dir : str
        the directory of the output path (must exist)
    """
    db = utils.init_h5_db(db_name+".h5", save_dir)

    # collect data
    for split in xrange(1, 6):
        (data_collector, value_collector, start_pos_collector,
         pos_traj_collector, goal_pos_collector) = sample_data_grid16(
            split, num_samples)
        group_name = "grid_data_split_"+str(split)
        utils.add_h5_group(group_name, db)
        utils.add_h5_ds(data_collector, "data", db, group_name)
        utils.add_h5_ds(value_collector, "value", db, group_name)
        with open(os.path.join(save_dir, db_name+"_start_%i.pkl" % (split)),
                  "w") as f:
            pickle.dump(start_pos_collector, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(os.path.join(save_dir, db_name+"_traj_%i.pkl" % (split)),
                  "w") as f:
            pickle.dump(pos_traj_collector, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(os.path.join(save_dir, db_name+"_goal_%i.pkl" % (split)),
                  "w") as f:
            pickle.dump(goal_pos_collector, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        print ("[MESSAGE] Save dataset %i at %s" % (split, save_dir))

    db.flush()
    db.close()
    print ("[MESSAGE] Save dataset at %s" % (save_dir))


def create_train_grid28(db_name, save_dir, num_samples=0):
    """Create training dataset for 28x28 grid.

    Parameters
    ----------
    db_name : str
        the name of dataset
    save_dir : str
        the directory of the output path (must exist)
    """
    db = utils.init_h5_db(db_name+".h5", save_dir)

    # collect data
    for split in xrange(5):
        dc_tot = None
        vc_tot = None
        spc_tot = []
        ptc_tot = []
        gpc_tot = []
        for idx in xrange(split*4+1, split*4+5):
            (data_collector, value_collector, start_pos_collector,
             pos_traj_collector, goal_pos_collector) = sample_data_grid28(
                idx, num_samples)
            if dc_tot is None:
                dc_tot = data_collector
            else:
                dc_tot = np.vstack((dc_tot, data_collector))
            if vc_tot is None:
                vc_tot = value_collector
            else:
                vc_tot = np.vstack((vc_tot, value_collector))
            spc_tot += start_pos_collector
            ptc_tot += pos_traj_collector
            gpc_tot += goal_pos_collector
        group_name = "grid_data_split_"+str(split+1)
        utils.add_h5_group(group_name, db)
        utils.add_h5_ds(dc_tot, "data", db, group_name)
        utils.add_h5_ds(vc_tot, "value", db, group_name)
        with open(os.path.join(save_dir, db_name+"_start_%i.pkl" % (split+1)),
                  "w") as f:
            pickle.dump(spc_tot, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(os.path.join(save_dir, db_name+"_traj_%i.pkl" % (split+1)),
                  "w") as f:
            pickle.dump(ptc_tot, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(os.path.join(save_dir, db_name+"_goal_%i.pkl" % (split+1)),
                  "w") as f:
            pickle.dump(gpc_tot, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        print ("[MESSAGE] Save dataset %i at %s" % (split+1, save_dir))

    db.flush()
    db.close()
    print ("[MESSAGE] Save dataset at %s" % (save_dir))


def create_train_grid40(db_name, save_dir, num_samples=0):
    """Create training dataset for 40x40 grid.

    Parameters
    ----------
    db_name : str
        the name of dataset
    save_dir : str
        the directory of the output path (must exist)
    """
    db = utils.init_h5_db(db_name+".h5", save_dir)

    # collect data
    for split in xrange(5):
        dc_tot = None
        vc_tot = None
        spc_tot = []
        ptc_tot = []
        gpc_tot = []
        for idx in xrange(split*20+1, split*20+21):
            (data_collector, value_collector, start_pos_collector,
             pos_traj_collector, goal_pos_collector) = sample_data_grid40(
                idx, num_samples)
            if dc_tot is None:
                dc_tot = data_collector
            else:
                dc_tot = np.vstack((dc_tot, data_collector))
            if vc_tot is None:
                vc_tot = value_collector
            else:
                vc_tot = np.vstack((vc_tot, value_collector))
            spc_tot += start_pos_collector
            ptc_tot += pos_traj_collector
            gpc_tot += goal_pos_collector
        group_name = "grid_data_split_"+str(split+1)
        utils.add_h5_group(group_name, db)
        utils.add_h5_ds(dc_tot, "data", db, group_name)
        utils.add_h5_ds(vc_tot, "value", db, group_name)
        with open(os.path.join(save_dir, db_name+"_start_%i.pkl" % (split+1)),
                  "w") as f:
            pickle.dump(spc_tot, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(os.path.join(save_dir, db_name+"_traj_%i.pkl" % (split+1)),
                  "w") as f:
            pickle.dump(ptc_tot, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(os.path.join(save_dir, db_name+"_goal_%i.pkl" % (split+1)),
                  "w") as f:
            pickle.dump(gpc_tot, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        print ("[MESSAGE] Save dataset %i at %s" % (split+1, save_dir))

    db.flush()
    db.close()
    print ("[MESSAGE] Save dataset at %s" % (save_dir))


def load_train_grid8(return_imsize=True):
    """Load train 8x8 grid."""
    file_base_path = os.path.join(rlvision.RLVISION_DATA,
                                  "train", "gridworld_8", "gridworld_8")

    # load dataset
    if not os.path.isfile(file_base_path+".h5"):
        raise ValueError("The dataset %s is not existed!" %
                         (file_base_path+".h5"))

    db = h5py.File(file_base_path+".h5", mode="r")

    with open(file_base_path+"_start.pkl", "r") as f:
        start_pos_list = pickle.load(f)
        f.close()

    with open(file_base_path+"_traj.pkl", "r") as f:
        traj_list = pickle.load(f)
        f.close()

    with open(file_base_path+"_goal.pkl", "r") as f:
        goal_list = pickle.load(f)
        f.close()

    if return_imsize is True:
        return (db['data'], db['value'], start_pos_list, traj_list,
                goal_list, (8, 8))
    else:
        return (db['data'], db['value'], start_pos_list, traj_list,
                goal_list)


def load_train_grid16(return_imsize=True):
    """Load train 16x16 grid."""
    file_base_path = os.path.join(rlvision.RLVISION_DATA,
                                  "train", "gridworld_16", "gridworld_16")

    if not os.path.isfile(file_base_path+".h5"):
        raise ValueError("The dataset %s is not existed!" %
                         (file_base_path+".h5"))

    db = h5py.File(file_base_path+".h5", mode="r")

    # load dataset
    data = None
    value = None
    start_tot = []
    traj_tot = []
    goal_tot = []
    for split in xrange(1, 6):
        if data is None:
            data = db["grid_data_split_"+str(split)]['data']
        else:
            data = np.vstack((data,
                              db["grid_data_split_"+str(split)]['data']))
        if value is None:
            value = db["grid_data_split_"+str(split)]['value']
        else:
            value = np.vstack((value,
                               db["grid_data_split_"+str(split)]['value']))

        with open(file_base_path+"_start_%i.pkl" % (split), "r") as f:
            start_pos_list = pickle.load(f)
            f.close()
        start_tot += start_pos_list

        with open(file_base_path+"_traj_%i.pkl" % (split), "r") as f:
            traj_list = pickle.load(f)
            f.close()
        traj_tot += traj_list

        with open(file_base_path+"_goal_%i.pkl" % (split), "r") as f:
            goal_list = pickle.load(f)
            f.close()
        goal_tot += goal_list

    if return_imsize:
        return data, value, start_tot, traj_tot, goal_tot, (16, 16)
    else:
        return data, value, start_tot, traj_tot, goal_tot


def load_train_grid28(return_imsize=True):
    """Load train 28x28 grid."""
    file_base_path = os.path.join(rlvision.RLVISION_DATA,
                                  "train", "gridworld_28", "gridworld_28")

    if not os.path.isfile(file_base_path+".h5"):
        raise ValueError("The dataset %s is not existed!" %
                         (file_base_path+".h5"))

    db = h5py.File(file_base_path+".h5", mode="r")

    # load dataset
    data = None
    value = None
    start_tot = []
    traj_tot = []
    goal_tot = []
    for split in xrange(1, 6):
        if data is None:
            data = db["grid_data_split_"+str(split)]['data']
        else:
            data = np.vstack((data,
                              db["grid_data_split_"+str(split)]['data']))
        if value is None:
            value = db["grid_data_split_"+str(split)]['value']
        else:
            value = np.vstack((value,
                               db["grid_data_split_"+str(split)]['value']))

        with open(file_base_path+"_start_%i.pkl" % (split), "r") as f:
            start_pos_list = pickle.load(f)
            f.close()
        start_tot += start_pos_list

        with open(file_base_path+"_traj_%i.pkl" % (split), "r") as f:
            traj_list = pickle.load(f)
            f.close()
        traj_tot += traj_list

        with open(file_base_path+"_goal_%i.pkl" % (split), "r") as f:
            goal_list = pickle.load(f)
            f.close()
        goal_tot += goal_list

    if return_imsize:
        return data, value, start_tot, traj_tot, goal_tot, (28, 28)
    else:
        return data, value, start_tot, traj_tot, goal_tot


def load_train_grid40(return_imsize=True):
    """Load train 40x40 grid."""
    file_base_path = os.path.join(rlvision.RLVISION_DATA,
                                  "train", "gridworld_40", "gridworld_40")

    if not os.path.isfile(file_base_path+".h5"):
        raise ValueError("The dataset %s is not existed!" %
                         (file_base_path+".h5"))

    db = h5py.File(file_base_path+".h5", mode="r")

    # load dataset
    data = None
    value = None
    start_tot = []
    traj_tot = []
    goal_tot = []
    for split in xrange(1, 6):
        if data is None:
            data = db["grid_data_split_"+str(split)]['data']
        else:
            data = np.vstack((data,
                              db["grid_data_split_"+str(split)]['data']))
        if value is None:
            value = db["grid_data_split_"+str(split)]['value']
        else:
            value = np.vstack((value,
                               db["grid_data_split_"+str(split)]['value']))

        with open(file_base_path+"_start_%i.pkl" % (split), "r") as f:
            start_pos_list = pickle.load(f)
            f.close()
        start_tot += start_pos_list

        with open(file_base_path+"_traj_%i.pkl" % (split), "r") as f:
            traj_list = pickle.load(f)
            f.close()
        traj_tot += traj_list

        with open(file_base_path+"_goal_%i.pkl" % (split), "r") as f:
            goal_list = pickle.load(f)
            f.close()
        goal_tot += goal_list

    if return_imsize:
        return data, value, start_tot, traj_tot, goal_tot, (40, 40)
    else:
        return data, value, start_tot, traj_tot, goal_tot
