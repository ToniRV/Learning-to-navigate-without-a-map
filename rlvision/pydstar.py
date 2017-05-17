"""Implement a replicate D* algorithm in Python.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import math
from collections import OrderedDict
import heapq


class State(object):
    """The state."""
    def __init__(self, x, y, k):
        """The state init."""
        self.x = x
        self.y = y
        self.k = k  # a list
        self.eps = 0.00001

    def equal(self, state):
        """compare if state equal."""
        if self.x == state.x and self.y == state.y:
            return True
        return False

    def gt(self, state):
        """Greater than."""
        if self.k[0]-self.eps > state.k[0]:
            return True
        elif self.k[0] < state.k[0]-self.eps:
            return False
        return self.k[1] > state.k[1]

    def le(self, state):
        """Less than or equal."""
        if self.k[0] < state.k[0]:
            return True
        elif self.k[0] > state.k[0]:
            return False
        return self.k[1] < state.k[1] + self.eps

    def lt(self, state):
        """Less than."""
        if self.k[0]+self.eps < state.k[0]:
            return True
        elif self.k[0]-self.eps > state.k[0]:
            return False
        return self.k[1] < state.k[1]


class iPoint2(object):
    """Represent a point."""
    def __init__(self, x, y):
        """point."""
        self.x = x
        self.y = y


class CellInfo(object):
    """Cell info."""
    def __init__(self, g, rhs, cost):
        """Cell info."""
        self.g = g
        self.rhs = rhs
        self.cost = cost


def state_hash(state):
    """State hash."""
    return state.x+34245*state.y


class Dstar(object):
    """A python D* algorithm"""
    def __init__(self,
                 start_x,
                 start_y,
                 goal_x,
                 goal_y,
                 max_steps=80000,
                 c1=1):
        """Init Dstar algo.

        Parameters
        ----------
        max_steps : int
            node expansions before we give up
        c1 : int
            cost of an unseen cell
        """
        self.max_steps = max_steps
        self.c1 = c1
        self.M_SQRT2 = math.sqrt(2)
        # clean cell hash
        self.cell_hash = OrderedDict()

        # clean path
        self.path = []

        # open hash

        # open list
        self.open_list = []

        # pop everything from open list

        self.k_m = 0
        self.s_start = State(start_x, start_y)
        self.s_goal = State(goal_x, goal_y)

        # cell hash goal
        self.cell_hash[self.s_goal] = CellInfo(0, 0, self.c1)

        # cell hash start
        temp_g = temp_rhs = self.heuristic(self.s_start, self.s_goal)
        self.cell_hash[self.s_start] = CellInfo(temp_g, temp_rhs, self.c1)

        self.s_start = self.calculate_key(self.s_start)

        self.s_last = self.s_start

    def update_cell(self, x, y, val):
        """update cell."""
        pass

    def update_start(self, x, y):
        """Update the position of the robot, this does not force a replan."""
        self.s_start.x = x
        self.s_start.y = y

        self.k_m += self.heuristic(self.s_last, self.s_start)

        self.s_start = self.calculate_key(self.s_start)

        self.s_last = self.s_start

    def update_goal(self, x, y):
        """update goal."""
        pass

    def replan(self):
        """replan."""
        pass

    def get_path(self):
        """get path."""
        return self.path

    def close(self, x, y):
        """Returns true if x and y are within 10E-5, false otherwise."""
        return float(abs(x-y)) < 0.00001

    def make_new_cell(self, u):
        """Checks if a cell is in the hash table, if not it adds it in."""
        if self.cell_hash[u] != self.cell_hash[-1]:
            return

        tmp = CellInfo(self.heuristic(u, self.s_goal),
                       self.heuristic(u, self.s_goal),
                       self.c1)
        self.cell_hash[u] = tmp

    def getG(self, u):
        """Get the G value for state u."""
        if self.cell_hash[u] == self.cell_hash[-1]:
            return self.heuristic(u, self.s_goal)
        return self.cell_hash[u].g

    def get_rhs(self, u):
        """Get rhs value for state u."""
        if u.equal(self.s_goal):
            return 0
        if self.cell_hash[u].equal():
            return self.heuristic(u, self.s_goal)
        return self.cell_hash[u].rhs

    def setG(self, u, g):
        """Sets the G value for state u."""
        self.make_new_cell(u)
        self.cell_hash[u].g = g

    def set_rhs(self, u, rhs):
        """Sets the rhs value for state u."""
        self.make_new_cell(u)
        self.cell_hash[u].rhs = rhs

    def eight_condist(self, a, b):
        """8 way distance between state a and b."""
        minimum = float(abs(a.x-b.x))
        maximum = float(abs(a.y-b.y))

        if minimum > maximum:
            minimum, maximum = maximum, minimum

        return (self.M_SQRT2-1.0)*minimum+maximum

    def compute_shorest_path(self):
        """Compute shortest path."""
        s = []

        if len(self.open_list) == 0:
            return 1

        k = 0
        pass

    def update_vertex(self, u):
        """update vertex."""
        s = []

        if not u.equal(self.s_goal):
            pass

    def insert(self, u):
        """insert."""
        pass

    def remove(self, u):
        """remove."""
        pass

    def true_dist(self, a, b):
        """true distance."""
        pass

    def heuristic(self, a, b):
        """Calculate Heuristic."""
        return self.eight_condist(a, b)*self.c1

    def calculate_key(self, u):
        """calculate key."""
        val = min(self.get_rhs(u), self.get_g(u))
        u.k[0] = val+self.heuristic(u, self.s_start)+self.km
        u.k[1] = val

        return u

    def get_succ(self, u, s):
        """Returns a list of successor states for state u."""
        s = []
        u.k[0] = -1
        u.k[1] = -1

        if self.occupied(u):
            return s

        u.x += 1
        s = [u]+s
        u.y += 1
        s = [u]+s
        u.x -= 1
        s = [u]+s
        u.x -= 1
        s = [u]+s
        u.y -= 1
        s = [u]+s
        u.y -= 1
        s = [u]+s
        u.x += 1
        s = [u]+s
        u.x += 1
        s = [u]+s

        return s

    def get_pred(self, u, s):
        """Returns a list of all the predecessor states for state u."""
        s = []
        u.k[0] = -1
        u.k[0] = -1

        u.x += 1
        if not self.occupied(u):
            s = [u]+s
        u.y += 1
        if not self.occupied(u):
            s = [u]+s
        u.x -= 1
        if not self.occupied(u):
            s = [u]+s
        u.x -= 1
        if not self.occupied(u):
            s = [u]+s
        u.y -= 1
        if not self.occupied(u):
            s = [u]+s
        u.y -= 1
        if not self.occupied(u):
            s = [u]+s
        u.x += 1
        if not self.occupied(u):
            s = [u]+s
        u.x += 1
        if not self.occupied(u):
            s = [u]+s

        return s

    def cost(self, a, b):
        """calculate cost."""
        pass

    def occupied(self, u):
        """returns true if the cell is occupied (non-traversable).

        Need review
        """
        if self.cell_hash[u] == self.cell_hash[-1]:
            return False
        return self.cell_hash[u].second.cost < 0

    def is_valid(self, u):
        """Checking if the state is valid.

        Returns true if state u is on the open list or not by checking if
        it is in the hash table.
        """
        pass

    def key_hash_code(self, u):
        """Return the key has code for the state u."""
        return float(u.k[0]+1193*u.k[1])
