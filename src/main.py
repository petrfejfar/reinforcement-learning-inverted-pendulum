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

import random
import os
import subprocess

from math import cos, sin, floor, pi
from sys import platform as _platform

import matplotlib.pyplot as plt

from Qtable import QTable

_DIR = os.path.dirname(os.path.realpath(__file__))


def simulate(force, state):
    """Compute the next states given the force and the current states"""
    x = state[0]
    x_dot = state[1]
    theta = state[2]
    theta_dot = state[3]

    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSPOLE + MASSCART
    LENGTH = 0.5
    POLEMASS_LENGTH = MASSPOLE * LENGTH
    STEP = 0.02
    FOURTHIRDS = 4.0 / 3

    costheta = cos(theta)
    sintheta = sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS

    thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS))

    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Update the four state variables, using Euler's method.

    return (x + STEP * x_dot,
            x_dot + STEP * xacc,
            theta + STEP * theta_dot,
            theta_dot + STEP * thetaacc)


def system_safe(state):
    """Is the system in stable?"""
    theta = state[2] - (2*pi) * floor((state[2] + pi) / (2*pi))
    return abs(state[0]) < 2.4 and abs(theta) < 0.20943951


def draw_state(state, force, filename):
    plt.clf()
    plt.xlim([-2.4, 2.4])
    plt.ylim([-1, 3.8])

    plt.title('state: %+.2f %+.2f %+.2f %+.2f' % state)
    if system_safe(state):
        color = "green"
    else:
        color = "red"

    # draw state
    plt.plot([state[0] + 0, state[0] + sin(state[2])], [0, cos(state[2])], color=color, aa=True)

    # draw 12 degree safezone
    twelve_rads = 0.20943951
    plt.plot([state[0] + 0, state[0] + sin(twelve_rads)], [0, cos(twelve_rads)], color="grey", aa=True)
    plt.plot([state[0] + 0, state[0] + sin(-twelve_rads)], [0, cos(-twelve_rads)], color="grey", aa=True)

    # draw force
    if force > 0:
        plt.plot([0, 1], [0, 0], color="black", aa=True)
    else:
        plt.plot([0, -1], [0, 0], color="black", aa=True)

    plt.savefig(filename)

def condition_based_action(state):
    force = 0

    if state[2] > 0:
        force = 10
        if state[2] < 0.2 and state[3] < -0.5:
            force = -10
    else:
        force = -10
        if state[2] > -0.2 and state[3] > 0.5:
            force = 10

    return force

def r_time(state):
    """Return 2 (reward for two milliseconds in safe state) or 0"""
    if system_safe(state):
        return 2
    else:
        return 0

def r_theta(state):
    """Return reward based on pole angle"""
    return state[2] ** 2

def main():
    """Main procedure"""

    # Init data structures
    qtable = QTable([-10, 10], [(-2.4, 2.4, 24), (-12, 12, 24), (-pi, pi, 120), (-pi, pi, 60)])

    # Reinforcement learning
    # TODO

    # Run inverted pendulum system simulation
    state = (0.0, 0.0, random.random() - 0.5, 0.0) # slightly skew pole as start state

    for i in range(0, 1000, 2):
        print("%.2fsec" % (i / 100.0), state)
        print(system_safe(state))

        force = condition_based_action(state)
        # force = qtable.get_best_action(state)

        state = simulate(force, state)
        draw_state(state, force, os.path.join(_DIR, "./../output/state_%03dms.png" % (i/2)))

    # Generate video
    if _platform == "linux":                                         # GNU/Linux
        subprocess.call(os.path.join(_DIR, "./../make_video.sh"), shell=True)
    elif _platform == "darwin":                                           # OS X
        pass
    elif _platform == "win32" or _platform == "cygwin":             # Windows...
        pass

if __name__ == '__main__':
    main()
