"""Run dstar given hdf5 data

Author: Antoni Rosinol
Email : tonirosinol
"""
import subprocess as sp
import os
import numpy as np

def run_dstar (start, goal, grid, imsize):
    """Run dstar algorithm: computes shortest
        path from start to goal given a grid with
        obstacles information.

    Parameters
    ----------
    start : (int, int)
        start position in (x, y) coordinates
    goal : (int, int)
        goal position in (x, y) coordinates
    grid : list of ints
        flattened square grid represented by a list
        of ints equal to either 0 or 1 (obstacles and
        free space respectively).
    imsize : int
        size of the grid sides. It corresponds to
        the grids width/height since the grid is
        squared.

    Returns
    -------
    updated_grid : list of ints
        flattened grid with a list of ints equal to
        0, 1, and other values representing obstacles,
        free space and the shortest path respectively.
    send_error : bool
        flag indicating whether a path between start
        and goal could be found or not (returns true
        no matter what the error is)
    """

    # Get start index
    start_index = np.ravel_multi_index(start, imsize, order='F')
    if grid[start_index] == 0:
        print("[ERROR] start position falls over an obstacle")
    else:
        # Color in grey the start position
        #TODO Copy grid and use updated_grid
        # Right now, do not set this to an int of 2 or more digits
        grid[start_index] = 1

    # Get goal index
    # TODO Get value data containing the reward values.
    # The database puts a number 10 wherever the goal is (I think)
    # value_data = db["value_data"]
    goal_index = np.ravel_multi_index(goal, imsize, order='F')
    if grid[goal_index] == 0:
        print("[ERROR] goal position falls over an obstacle")
    else:
        # Color in grey the start position
        #TODO Copy grid and use updated_grid
        # Right now, do not set this to an int of 2 or more digits
        grid[goal_index] = 1

    # Get current working directory
    # TODO I don't know if there is a better way?
    dir = os.getcwd()
    exe_path = dir+"/dstar-lite/build/dstar_from_input"
    if not os.path.isfile(exe_path):
        raise ValueError("The executable %s does not exist!" % (exe_path))

    # Utility function.
    stringify = lambda x: str(x).strip('[()]').replace(',', '').replace(' ', '').replace('\n', '')

    # Run dstar algorithm in c++
    # Send start_index, goal_index, size of the grid and the grid through std input
    # All inputs must be flattened, aka string of int or ints (no matrices)
    # I.e. a 2x2 grid would be given as a string of 4 ints (row-major, C style).
    dstar_subprocess = sp.Popen([exe_path,
                                stringify(start_index), stringify(goal_index),
                                stringify(grid)],
                                stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

    # Answer from dstar algorithm.
    # It is send through stdout but also catches stderr.
    response = dstar_subprocess.communicate()

    # Response is given as (stdout , stderr)
    answer = response[0].splitlines()
    errors = response[1].splitlines()

    send_error = False
    if len(errors) == 0:
        for a in answer:
            grid[int(a)] = 150
    else:
        print("[ERROR] Errors found while running dstar algorithm.")
        print(errors)
        send_error = True

    return grid, send_error
