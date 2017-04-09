"""Implement a replicate D* algorithm in Python.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import math


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
        self.cell_hash = {}

        # clean path

        # open hash

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

    def calculate_key(self, u):
        """calculate key."""
        val = min(self.get_rhs(u), self.get_g(u))
        u.k[0] = val+self.heuristic(u, self.s_start)+self.km
        u.k[1] = val

        return u

    def get_rhs(self, u):
        """Get rhs value for state u."""
        if u.equal(self.s_goal):
            return 0
        if self.cell_hash[u].equal():
            return self.heuristic(u, self.s_goal)
        return self.cell_hash[u].rhs

    def heuristic(self, a, b):
        """Calculate Heuristic."""
        return self.eight_condist(a, b)*self.c1

    def eight_condist(self, a, b):
        """8 way distance between state a and b."""
        minimum = float(abs(a.x-b.x))
        maximum = float(abs(a.y-b.y))

        if minimum > maximum:
            minimum, maximum = maximum, minimum

        return (self.M_SQRT2-1.0)*minimum+maximum

    def key_hash_code(u):
        """Return the key has code for the state u."""
        return float(u.k[0]+u.k[1])
