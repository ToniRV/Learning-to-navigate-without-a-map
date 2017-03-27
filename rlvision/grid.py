"""Grid class to have a uniform grid functions.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import numpy as np

import rlvision
from rlvision import utils


class Grid(object):
    """A centralized class for managing grid."""

    def __init__(self, grid_map, value_map, im_size=None,
                 start_pos=None, is_po=True, mask_radius=3,
                 grid_type="one-is-free"):
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

    def update_curr_map(self, map_update):
        """Update current map.

        Parameters
        ----------
        map_update : numpy.ndarray
            the update to the current map
        """
        self.curr_map = utils.accumulate_map(self.curr_map, map_update,
                                             one_is_free=self.one_is_free)

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
            self.update_curr_map(self.get_curr_visible_map(pos_update))
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

    def get_state_reward(self):
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
        recent_pos = self.pos_history[-1]
        if recent_pos == self.goal_pos and \
           self.get_time() <= self.im_size[0]+self.im_size[1]:
            # success
            return self.value_map[recent_pos[0], recent_pos[1]], 1
        elif self.get_time() > self.im_size[0]+self.im_size[1]:
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
        grid = self.grid_data[self.curr_idx]
        value = self.value_data[self.curr_idx]
        goal_pos = self.get_goal_pos(value)
        print ("[MESSAGE] GOAL POSITION:", goal_pos)

        # find end block idx
        curr_idx = self.curr_idx
        while np.array_equal(grid, self.grid_data[curr_idx]):
            curr_idx += 1

        # parse grid traj
        start_pos_list = []
        pos_traj = []
        flag = True
        for idx in xrange(self.curr_idx, curr_idx):
            curr_pos = (self.states_xy[idx][0], self.states_xy[idx][1])
            next_pos = self.get_next_state(curr_pos, self.label_data[idx])

            if flag is True:
                # when find a new start
                print ("[MESSAGE] A START POINT:", curr_pos)
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

            print (curr_pos, next_pos, self.label_data[idx])
            temp_pos_traj.append(curr_pos)

        # update current idx
        self.curr_idx = curr_idx

        return grid, value, start_pos_list, pos_traj
