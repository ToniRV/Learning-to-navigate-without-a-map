#!/usr/bin/env python
from __future__ import print_function

import sys

import os
import cPickle as pickle
import numpy as np

from math import pi, atan, tan, cos, acos, sin, asin
from xml.etree import ElementTree
import copy

import rlvision
from rlvision import utils
from rlvision.grid import GridDataSampler, Grid
from rlvision.dstar import Dstar

class world_model:
    def __init__ (self):
        # create separate trees for the empty world and for models resource
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(self.dir_path + "/../world_models/default_world.world", 'rt') as f:
            self.world_tree = ElementTree.parse(f)
        with open(self.dir_path + "/../world_models/default_models.world", 'rt') as f:
            self.models_tree = ElementTree.parse(f)

        self.default_box_node = 0
        self.default_cone     = 0
        self.default_tarmac   = 0
        # Set world node to point to the empty world tree
        self.world_node = self.world_tree.find("world")

        # Fetch models from the models tree
        for model in self.models_tree.find("world").findall("model"):
            if model.attrib["name"] == "unit_box":
                self.default_box_node = copy.deepcopy(model)
            if model.attrib["name"] == "Construction Barrel":
                self.default_cone = copy.deepcopy(model)
            if model.attrib["name"] == "asphalt_plane":
                self.default_tarmac = copy.deepcopy(model)
            if model.attrib["name"] == "grey_wall":
                self.default_wall = copy.deepcopy(model)

    def add_point(self, x, y):
        model = copy.deepcopy(self.default_box_node)
        link_node     = model.find("link")
        pose_node     = link_node.find("pose")
        pose_node.text= str(x) + " " + str(y) + " 0 0 0 0"
        self.world_node.append(model)

    def add_cone(self, x, y):
        model = copy.deepcopy(self.default_cone)
        pose_node     = model.find("pose")
        pose_node.text= str(x) + " " + str(y) + " 0 0 0 0"
        self.world_node.append(model)

    def add_tarmac(self, x, y, theta, length, width):
        model = copy.deepcopy(self.default_tarmac)
        pose_node      = model.find("pose")
        pose_node.text = str(x) + " " + str(y) + " 0 0 0 " + str(theta)
        size_node = model.find("link").find("visual").find("geometry").find("box").find("size")
        size_node.text = str(length) + " " + str(width) + " 0.1"
        self.world_node.append(model)

    def add_wall(self, x, y, theta, length, width):
        model = copy.deepcopy(self.default_wall)
        pose_node      = model.find("pose")
        pose_node.text = str(x) + " " + str(y) + " 0 0 0 " + str(theta)
        size_node = model.find("link").find("visual").find("geometry").find("box").find("size")
        size_node.text = str(length) + " " + str(width) + " 2"
        self.world_node.append(model)

    def stringToXml(self, string):
        print("STRING")
        children = ElementTree.fromstring(string.data)
        self.world_node.append(children)

    def create_world_from_grid(self, grid, im_size):
        # Create gazebo world out of the grid
        scale = 0.75

        # Build walls around the field
        wall_width = 0.5
        self.add_wall(scale*(im_size[0]-1)/2.0, 0, 0, scale*(im_size[0]-1), wall_width)
        self.add_wall(0, scale*(im_size[1]-1)/2.0, pi / 2.0, scale*(im_size[0]-1), wall_width)
        self.add_wall(scale*(im_size[0]-1), scale*(im_size[1]-1)/2.0, - pi / 2.0, scale*(im_size[0]-1), wall_width)
        self.add_wall(scale*(im_size[0]-1)/2.0, scale*(im_size[1]-1), pi, scale*(im_size[0]-1), wall_width)

        # Add asphalt
        self.add_tarmac(scale*(im_size[0]-1)/2.0, scale*(im_size[1]-1)/2.0, 0, scale*(im_size[0]-1), scale*(im_size[1]-1))

        # Add cones wherever there should be obstacles
        i = 1
        j = 1
        obstacle_indices = np.where(grid != 1)
        unraveled_indices = np.unravel_index(obstacle_indices, im_size, order='C')
        for x in grid:
            if (grid[j+i*im_size[0]] != 1):
                self.add_cone(scale*j, scale*i)
                self.write()
            j += 1
            if (j % (im_size[1]-1)) == 0:
                j = 1
                i +=1
                if (i == im_size[0]-1):
                    break

    def write(self):
	    self.world_tree.write(self.dir_path + "/../world_models/python_generated.world", encoding='utf-8')

def main(args):
    # setup result folder
    model_name = "dstar-16"
    model_path = os.path.join(rlvision.RLVISION_MODEL, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    print ("[MESSAGE] The model path is created at %s" % (model_path))

    # load data
    db, im_size = utils.load_grid16(split=1)

    # prepare relevant data
    im_data = db['im_data']
    value_data = db['value_data']
    states = db['state_xy_data']
    label_data = db['label_data']
    print ("[MESSAGE] The data is loaded.")
    print ("[MESSAGE] Get data sampler...")
    grid_sampler = GridDataSampler(im_data, value_data, im_size, states,
                                   label_data)
    print ("[MESSAGE] Data sampler ready.")

    # sample grid
    print ("[MESSAGE] SAMPLING NEW GRID")
    grid, value, start_pos_list, pos_traj, goal_pos = grid_sampler.next()
    print ("[MESSAGE] New Grid is sampled.")
    print ("[MESSAGE] Number of trajectories:", len(start_pos_list))

    utils.plot_grid(grid, im_size)

    # Create gazebo world from grid
    wm = world_model()
    wm.create_world_from_grid(grid, im_size)

if __name__ == '__main__':
    main(sys.argv)
