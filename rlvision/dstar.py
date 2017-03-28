"""Run dstar given hdf5 data

Author: Antoni Rosinol
Email : tonirosinol
"""
import subprocess as sp
import os
import numpy as np
import zmq


class Dstar:
    """A central class to carry out D* algorithm with binary."""
    def add_obstacle(self, x, y):
        msg = str(x)+" "+str(y)

        # Request a cell update
        print("[INFO] Sending cell update request")
        self.socket.send(b"update")

        #  Get the reply.
        errors = False
        if (self.socket.recv() != "go"):
            print("[ERROR] Socket could not process update request.")
            errors = True
            self.socket.send(b"")
        else:
            self.socket.send(msg)

        if (self.socket.recv() != "ok"):
            print("[ERROR] Socket was not able to update given cell.")
            errors = True
        else:
            index = np.ravel_multi_index((x, y), self.imsize, order='C')
            print("[INFO] Updating new obstacle")
            self.grid[index] = 0

        return errors

    def replan(self, next_move_only=True):
        # Request a replan
        print("[INFO] Sending replanning request")
        self.socket.send(b"replan")

        #  Get the reply.
        path = self.socket.recv()

        errors, solution_path = self.__process_path__(path, next_move_only)

        return errors, solution_path

    def kill_subprocess(self):
        # Request subprocess dead
        self.socket.send(b"kill")

        #  Get the reply.
        if (self.socket.recv() == "ok"):
            print("[INFO] Subprocess confirmed dead.")
        else:
            print("[ERROR] Subprocess did not respond to kill request")

    def reset_start_pos(self, start):
        """Reset a start pos.

        The position is always valid as it's calculated from
        the path before.

        Parameters
        ----------
        start : tuple
            the new start
        """
        self.start = start
        msg = str(start[0])+" "+str(start[1])
        # Request a cell update
        print("[INFO] Sending cell update request")
        self.socket.send(b"setstart")

        #  Get the reply.
        errors = False
        if (self.socket.recv() != "go"):
            print("[ERROR] Socket could not process update request.")
            errors = True
            self.socket.send(b"")
        else:
            self.socket.send(msg)

        if (self.socket.recv() != "ok"):
            print("[ERROR] Socket was not able to update given start.")
            errors = True
        else:
            print("[INFO] New start updated")

        return errors

    def __spawnDstar(self, start, goal, grid, imsize):
        """Spawn dstar algorithm: computes shortest
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
        """
        # Get start index
        start_index = np.ravel_multi_index(start, imsize, order='C')
        if grid[start_index] == 0:
            print("[ERROR] start position falls over an obstacle")
        else:
            # Color in grey the start position
            # TODO Copy grid and use updated_grid
            # Right now, do not set this to an int of 2 or more digits
            grid[start_index] = 1

        # Get goal index
        # TODO Get value data containing the reward values.
        # The database puts a number 10 wherever the goal is (I think)
        # value_data = db["value_data"]
        goal_index = np.ravel_multi_index(goal, imsize, order='C')
        if grid[goal_index] == 0:
            print("[ERROR] goal position falls over an obstacle")
        else:
            # Color in grey the start position
            # TODO Copy grid and use updated_grid
            # Right now, do not set this to an int of 2 or more digits
            grid[goal_index] = 1

        # Get current working directory
        # TODO I don't know if there is a better way?
        dir = os.getcwd()
        exe_path = dir+"/dstar-lite/build/dstar_from_input"
        if not os.path.isfile(exe_path):
            raise ValueError("The executable %s does not exist!" % (exe_path))

        # Utility function.
        stringify = lambda x: str(x).strip('[()]').replace(
            ',', '').replace(' ', '').replace('\n', '')

        # Run dstar algorithm in c++
        # Send start_index, goal_index, size of the grid and the grid
        # through std input
        # All inputs must be flattened, aka string of int or ints (no matrices)
        # I.e. a 2x2 grid would be given as a string of 4 ints
        # (row-major, C style).
        self.dstar_subprocess = sp.Popen(
            [exe_path, stringify(start_index), stringify(goal_index),
             stringify(grid)],
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

    def __process_path__(self, path, next_move_only=True):
        if len(path) != 0:
            path = path[:-1]
            print("[INFO] Received path: %s" % (path))
            # Clear last path
            #  for idx, value in enumerate(self.grid):
            #      if value == 150:
            #          self.grid[idx] = 1
            # Print new path
            path_list = []
            for a in path.split('.'):
                path_list.append(int(a))
                #  self.grid[int(a)] = 150
            path_list = np.unravel_index(path_list, self.imsize)
            solution_list = []
            for i in xrange(path_list[0].shape[0]):
                solution_list.append((path_list[0][i],
                                      path_list[1][i]))
            if next_move_only:
                return False, solution_list[1]
            else:
                return False, solution_list
        else:
            print("[ERROR] Errors found while running dstar algorithm.")
            return True

    def __init__(self, start, goal, grid, imsize):
        #  Socket to talk to server
        self.context = zmq.Context()
        print("[INFO] Connecting to dstar server")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        self.start = start
        self.goal = goal
        self.grid = grid
        self.imsize = imsize

        self.__spawnDstar(start, goal, grid, imsize)

    def __del__(self):
        print("[INFO] Killing subprocess")
        self.socket.close()
        self.context.term()
        self.dstar_subprocess.kill()
        response = self.dstar_subprocess.communicate()
        print("[LOG] Subprocess logged cout:\n"+response[0])
        print("[LOG] Subprocess logged cerr:\n"+response[1])
        print ("[INFO] Clean up done")
