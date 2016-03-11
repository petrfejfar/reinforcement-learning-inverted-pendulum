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
import copy
import glob
import os
import matplotlib.pyplot as plt
import subprocess

from math import cos, sin
from sys import platform as _platform

from CartPole import CartPole
from Agents import CartPoleQAgent

_DIR = os.path.dirname(os.path.realpath(__file__))

def draw_state(state, force, filename, qTable):
    plt.clf()
    plt.xlim([-2.4, 2.4])
    plt.ylim([-1, 3.8])

    plt.title('state: %+.2f %+.2f %+.2f %+.2f' % (state.x, state.xdot, state.theta, state.thetadot) + " %d" % state.getScore())
    if not state.failure():
        color = "green"
    else:
        color = "red"

    qVals = qTable[str(state)]
    plt.annotate("%09.2f" % qVals[-10], (-2, 3), color="red" if force > 0 else "green")
    plt.annotate("%09.2f" % qVals[10], (1.5, 3), color="green" if force > 0 else "red")
    # draw state
    plt.plot([state.x + 0, state.xdot + sin(state.theta)], [0, cos(state.thetadot)], color=color, aa=True)

    # draw 12 degree safezone
    twelve_rads = 0.20943951
    plt.plot([state.x + 0, state.x + sin(twelve_rads)], [0, cos(twelve_rads)], color="grey", aa=True)
    plt.plot([state.x + 0, state.x + sin(-twelve_rads)], [0, cos(-twelve_rads)], color="grey", aa=True)

    # draw force
    if force > 0:
        plt.plot([0, 1], [0, 0], color="black", aa=True)
    else:
        plt.plot([0, -1], [0, 0], color="black", aa=True)

    plt.savefig(filename)

def main():

    agent = CartPoleQAgent(epsilon=0.05, gamma=1, alpha=0.1, numTraining=10000)

    for x in range(agent.numTraining):
        state = CartPole()
        agent.registerInitialState(state)

        i = 0
        while not state.failure():
            i+=1
            observation = agent.observationFunction(copy.deepcopy(state))
            action = agent.getAction(observation)
            state = state.generateSuccessor(action)
        # print(i)
        agent.final(state)
    print("Length of last sequence ", i)

    # Clear after previous run
    for f in glob.glob(os.path.join(_DIR, "./../output/state_*ms.png")):
        try:
            if os.path.isfile(f):
                os.unlink(f)
        except Exception as e:
            print(e)

    # Simulate
    state = CartPole()

    for i in range(0, 30000, 2):
        print("%.2fsec" % (i / 100.0), "%+.4f %+.4f %+.8f %+.4f" % (state.x, state.xdot, state.theta, state.thetadot), agent.qTable[str(state)])
        print(not state.failure())


        action = agent.getPolicy(state)
        if action is None:
            print("FAIL")
            break
        state = state.generateSuccessor(action)

        draw_state(state, action, os.path.join(_DIR, "./../output/state_%03dms.png" % (i/2)), agent.qTable)

    # Generate video
    if _platform == "linux":                                         # GNU/Linux
        subprocess.call(os.path.join(_DIR, "./../make_video.sh"), shell=True)
    elif _platform == "darwin":                                           # OS X
        pass
    elif _platform == "win32" or _platform == "cygwin":             # Windows...
        pass


if __name__ == '__main__':
    main()
