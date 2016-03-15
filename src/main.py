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

import glob
import os
import subprocess

from sys import platform as _platform
from math import pi

from Qtable import QTable
from model import Model

_DIR = os.path.dirname(os.path.realpath(__file__))
INF = 100000

# model paramaters
AREA_SIZE = 2.4
SAFE_ANGLE_RAD = 12 * pi / 180.0  # 12 degree
SIMULATION_TIME_DELTA = 0.02

# learning parameters
LEARNING_ITERATION = 600
MAX_STATE_TRANSITIONS = 30000


def delete_temp_files():
    # Clear after previous run
    for f in glob.glob(os.path.join(_DIR, "./../output/state_*ms.png")):
        try:
            if os.path.isfile(f):
                os.unlink(f)
        except Exception as e:
            print(e)


def main():
    """Main procedure"""

    # Init data structures
    # NOTE: previous result:
    # fifty_rads = deg_to_rad(50)
    # qtable = QTable([-10, 10], [(-2.4, 2.4, 4), (-2, 2, 30), (-twelve_rads, twelve_rads, 96), (-fifty_rads, fifty_rads, 25)]) # inf - nahoda
    # qtable = QTable([-10, 10, -30, 30], [(-2.4, 2.4, 16), (-2, 2, 20), (-twelve_rads, twelve_rads, 48), (-fifty_rads, fifty_rads, 12)]) # 48s - nahoda
    qtable = QTable([-10, 10], [(-AREA_SIZE, AREA_SIZE, 8), (-1, 1, 10), (-SAFE_ANGLE_RAD, SAFE_ANGLE_RAD, 28), (-0.5, 0.5, 28)])

    # inverted pendulum model
    initial_state = (0.0, 0.0, 0.0, 0.0)
    model = Model(initial_state, AREA_SIZE, SAFE_ANGLE_RAD)

    # Reinforcement learning
    qtable.learn(model, LEARNING_ITERATION, MAX_STATE_TRANSITIONS, SIMULATION_TIME_DELTA)

    qtable.draw("qtable.png")

    delete_temp_files()

    # Run inverted pendulum system simulation
    model.reset()

    for i in range(0, 30001, 2):
        print("%.2fsec" % (i / 100.0), model.get_state(), qtable.get_q_vals(model.get_state()))
        print(model.system_safe())

        if not model.system_safe():
            print("FAIL")
            break

        force = qtable.get_best_action(model.get_state())
        model.simulate(force, SIMULATION_TIME_DELTA)

        model.draw_state(force, os.path.join(_DIR, "./../output/state_%06dms.png" % (i/2)), qtable)

    # Generate video
    if _platform == "linux":                                         # GNU/Linux
        subprocess.call(os.path.join(_DIR, "./../make_video.sh"), shell=True)
    elif _platform == "darwin":                                           # OS X
        pass
    elif _platform == "win32" or _platform == "cygwin":             # Windows...
        subprocess.call(os.path.join(_DIR, "./../make_video.bat"), shell=True)
        import winsound
        freq = 2500
        dur = 1000
        winsound.Beep(freq, dur)

if __name__ == '__main__':
    main()
