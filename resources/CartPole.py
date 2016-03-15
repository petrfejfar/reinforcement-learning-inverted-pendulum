#!/usr/bin/env python3
"""
Copyright (c) 2016 Petr Fejfar, Martin Vana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import copy, math

class CartPole:
    gravity = 9.8

    def __init__(self, x = 0.0, xdot = 0.0, theta = 0.0, thetadot = 0.0, masscart = 1.0, masspole = 0.1, polelength = 1):
        self.x = x
        self.xdot = xdot
        self.theta = theta
        self.thetadot = thetadot

        self.masscart = masscart
        self.masspole = masspole
        self.length = polelength / 2.0  # actually half of the the pole's length


        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02		  # seconds between state updates

        self.score = 0

    def getNormalizedState(self, state):
        return (
            round(state.x),
            round(state.xdot*2)/2,
            round(state.theta*5)/5,
            round(state.thetadot*2)/2
        )

    def getScore(self):
        return self.score

    def __str__( self ):
        return str(self.getNormalizedState(self))

    def failure(self):
        twelve_degrees = 12 * math.pi / 180.0
        if (not -2.4 < self.x < 2.4) or (not -twelve_degrees < self.theta < twelve_degrees):
            return True
        else:
            return False

    def reward(self):
        if self.failure():
            return 0
        else:
            return 1
        # if self.failure():
        #     return -1.0
        # else:
        #     return 0.0

    def generateSuccessor(self, action):
        force = action

        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        fourthirds = 4.0 / 3

        tmp = (force + self.polemass_length * (self.thetadot ** 2) * sintheta) / self.total_mass;
        thetaacc = (self.gravity * sintheta - costheta * tmp) / (self.length * (fourthirds - self.masspole * costheta ** 2 / self.total_mass))
        xacc = tmp - self.polemass_length * thetaacc * costheta / self.total_mass

        successor = copy.deepcopy(self)

        successor.x += self.tau * self.xdot
        successor.xdot += self.tau * xacc
        successor.theta += self.tau * self.thetadot
        successor.thetadot += self.tau * thetaacc

        successor.score += self.reward()

        return successor

    def getPossibleActions(self):
        """
        Returns all possible actions.
        """
        return [-self.force_mag, self.force_mag]

    def getLegalActions(self):
        """
        Returns the legal actions for the agent specified.
        """
        if self.failure(): return []

        return self.getPossibleActions()
