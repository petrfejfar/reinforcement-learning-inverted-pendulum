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

from math import cos, sin, floor, pi
import matplotlib.pyplot as plt

class Model:
    """Cart pole model"""

    gravity = 9.8

    def __init__(self, initial_state, area_size, safe_angle_rad, masscart=1.0, \
                 masspole=0.1, polelength=1):
        """Initialize model"""

        # Physical model
        self.state = initial_state  # [0] x, [1] x_dot, [2] theta, [3] theda_dot

        self.masscart = masscart
        self.masspole = masspole
        self.length = polelength / 2.0  # actually half of the the pole's length

        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

        # Constraints
        self.area_size = area_size
        self.safe_angle_rad = safe_angle_rad

    def reset(self, state=(0.0, 0.0, 0.0, 0.0)):
        """Reset model to a default state"""

        self.state = state

    def get_state(self):
        """Return state"""

        return self.state

    def simulate(self, force, timespan):
        """
            Compute the next states given the force and the current states
            Update the four state variables, using Euler's method.
        """
        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]

        costheta = cos(theta)
        sintheta = sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta)
        temp /= self.total_mass

        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3 - self.masspole * costheta * \
                                             costheta / self.total_mass))

        xacc = temp - self.polemass_length * thetaacc * costheta
        xacc /= self.total_mass

        self.state = ( \
                        x + timespan * x_dot, \
                        x_dot + timespan * xacc, \
                        theta + timespan * theta_dot, \
                        theta_dot + timespan * thetaacc \
                     )

    def reward(self):
        """Return reward model reward delta for recent state"""

        return self.r_time()

    def is_system_safe(self):
        """Is the system stable?"""

        return self.is_state_safe(self.state)

    def is_state_safe(self, state):
        """Is the state stable?"""

        x = state[0]
        # Normalize angle to <-pi, pi>
        theta = state[2] - (2*pi) * floor((state[2] + pi) / (2*pi))

        return abs(x) < self.area_size and abs(theta) < self.safe_angle_rad

    def r_time(self):
        """Return 1 if in safe state, 0 otherwise"""

        if self.is_system_safe():
            return 1
        else:
            return 0

    def draw_state(self, force, filename, qtable):
        """Draw image based on recent state"""

        plt.clf()
        plt.xlim([-self.area_size, self.area_size])
        plt.ylim([-1, 3.8])

        plt.title('state: %+.2f %+.2f %+.2f %+.2f' % self.state)

        if self.is_system_safe():
            color = "green"
        else:
            color = "red"

        qvals = qtable.get_q_vals(self.state)
        plt.annotate("%09.2f" % qvals[0], (-2, 3), \
                     color="red" if force > 0 else "green")
        plt.annotate("%09.2f" % qvals[1], (1.5, 3), \
                     color="green" if force > 0 else "red")

        x = self.state[0]
        theta = self.state[2]

        # draw state
        plt.plot([x, x + sin(theta)], [0, cos(theta)], color=color, aa=True)

        # draw safezone
        plt.plot([x, x + sin(self.safe_angle_rad)],
                 [0, cos(self.safe_angle_rad)], color="grey", aa=True)
        plt.plot([x, x + sin(-self.safe_angle_rad)], \
                 [0, cos(-self.safe_angle_rad)], color="grey", aa=True)

        # draw force
        if force > 0:
            plt.plot([0, 1], [0, 0], color="black", aa=True)
        else:
            plt.plot([0, -1], [0, 0], color="black", aa=True)

        plt.savefig(filename)
