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
                 start_pos=None, is_po=True,
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
        grid_type : string
            "one-is-free"
            "zero-is-free"
        """
        if grid_type == "one_is_free":
            self.one_is_free = True
            self.empty_value = 1
            self.ob_value = 0  # obstacle value in map
        elif grid_type == "zero_is_free":
            self.one_is_free = False
            self.empty_value = 0
            self.ob_value = 1
        else:
            raise ValueError("The grid type should be either 'one-is-free'"
                             " or 'zero-is-free'")

        # set if it's partially observable
        self.is_po = is_po

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
            if self.im_size[0]*self.im_size[1] == self.value_map.shape[0]:
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
            self.start_pos = self.rand_start_pos()
        else:
            self.start_pos = start_pos

        # set current agent position
        self.set_curr_pos(self.start_pos)

        # set initial history, a list, ordered
        self.pos_history = [self.start_pos]

        # set goal position, based on value map
        self.set_goal_pos()

    def is_pos_valid(self, pos):
        """Check if the position is valid."""
        assert isinstance(pos, tuple)

        if self.grid_map[pos[0], pos[1]] == self.empty_value:
            return True
        else:
            return False

    def rand_start_pos(self):
        pass

    def set_start_pos(self, start_pos):
        """Set the start position."""
        if self.is_pos_valid(start_pos):
            self.start_pos = start_pos
            # clear all the history caches
            self.set_curr_pos(start_pos)
            self.pos_history = [start_pos]
        else:
            print ("[MESSAGE] WARNING: The position is not valid, nothing"
                   " changes.")

    def set_goal_pos(self):
        """Set goal position based on the value map."""
        pass

    def set_curr_pos(self, curr_pos):
        if self.is_pos_valid(curr_pos):
            self.curr_pos = curr_pos
        else:
            print ("[MESSAGE] WARNING: The position is not a vaild point")

    def set_curr_map(self, curr_map):
        pass

