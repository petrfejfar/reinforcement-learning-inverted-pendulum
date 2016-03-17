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
import sys

from sys import platform as _platform
from math import pi

from qtable import QTable
from model import Model

_DIR = os.path.dirname(os.path.realpath(__file__))

# Model paramaters
AREA_SIZE = 2.4                                                      # in meters
SAFE_ANGLE_RAD = 12 * pi / 180.0                                    # 12 degrees

# Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1
LEARNING_ITERATION = 500
MAX_STATE_TRANSITIONS = 30000

# Simulation parameters
SIMULATION_TIME = 60 * 60                                           # in seconds
SIMULATION_TIME_DELTA = 0.02                                        # in seconds

def delete_temp_files():
    """Clear simulation images after previous run"""

    for img in glob.glob(os.path.join(_DIR, "./../output/state_*ms.png")):
        try:
            if os.path.isfile(img):
                os.unlink(img)
        except OSError as exception:
            print(exception)

def main():
    """Main procedure"""

    # Init data structures
    qtable = QTable([-10, 10], \
                    [ \
                        (-AREA_SIZE, AREA_SIZE, 8), \
                        (-1, 1, 10), \
                        (-SAFE_ANGLE_RAD, SAFE_ANGLE_RAD, 28), (-1, 1, 28) \
                    ])

    # Inverted pendulum model
    initial_state = (0.0, 0.0, 0.0, 0.0)
    model = Model(initial_state, AREA_SIZE, SAFE_ANGLE_RAD)

    # Reinforcement learning
    qtable.learn(model,
                 LEARNING_ITERATION, \
                 MAX_STATE_TRANSITIONS, \
                 SIMULATION_TIME_DELTA, \
                 LEARNING_RATE, \
                 DISCOUNT_FACTOR)

    # Visualize QTable
    qtable.draw(os.path.join(_DIR, "./../output/qtable.png"))

    # Clear temporary files
    delete_temp_files()

    # Run inverted pendulum system simulation
    model.reset()

    for i in range(0, round(SIMULATION_TIME / SIMULATION_TIME_DELTA) + 1):
        print("%.2fsec" % (i * SIMULATION_TIME_DELTA), \
              model.get_state(), \
              qtable.get_q_vals(model.get_state()))
        print(model.is_system_safe())

        if not model.is_system_safe():
            print("FAIL")
            break

        force = qtable.get_best_action(model.get_state())
        model.simulate(force, SIMULATION_TIME_DELTA)

        model.draw_state(force, os.path.join(_DIR, \
                                             "./../output/state_%06dms.png" % \
                                             (i)), qtable)

    # Generate video
    print("\nGenerating video...", end="")
    sys.stdout.flush()

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

    print("DONE")

if __name__ == '__main__':
    main()
