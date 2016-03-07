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

import itertools
import random

class QTable:
    iterations = 1000
    state_transitions = 500

    def __init__(self, actions, states):
        """
            actions [-10, 10]
            states [(min, max, inteval_counts),...]
        """
        self._gamma = 1
        self._actions = list(range(len(actions)))
        print (self._actions)
        self._action_map = actions
        self._state_descriptions = states

        state_len = len(states)
        values = []
        # state_cnt = 0

        for i in range(state_len):
            delta = (states[i][1] - states[i][0]) / float(states[i][2])
            values.append([])
            for x in range(states[i][2] + 1):
                values[i].append(states[i][0] + x * delta)
            print(values[i])
            # state_cnt = len(values[i])

        # self._

        all_states = list(itertools.product(*values))
        # val = [values[0], values[1]]
        # all_states = list(itertools.product(*val))
        # print("")
        # print("")
        # print("")
        # print(all_states)

        self._table = dict((el, [0,0]) for el in all_states)


    def get_best_action(self, s):
        state_norm = self._normalize_state(s)
        return self._action_map[self._table[state_norm].index(max(self._table[state_norm]))]

    def learn(self, simulate_f, reward_f):
        for i in range(1, self.iterations+1):
            print("Iteration #%05d" % i)

            state = (0.0, 0.0, 0.0, 0.0)
            state_norm = self._normalize_state(state)

            for x in range(self.state_transitions):
                a = self._get_random_action(state_norm)
                r = reward_f(state)

                next_s = simulate_f(self._action_map[a], state)
                next_s_norm = self._normalize_state(next_s)

                self._update_Q(state_norm, a, next_s_norm, r)

                state = next_s
                state_norm = self._normalize_state(state)

    def _get_random_action(self, s):
        return random.choice(self._actions)

    def _update_Q(self, s, a, s1, r):
        self._table[s][a] = r + self._gamma * max(self._table[s1])

    def _normalize_state(self, s):
        state_len = len(self._state_descriptions)

        state_norm = []
        for i in range(state_len):
            delta = (self._state_descriptions[i][1] - self._state_descriptions[i][0]) / float(self._state_descriptions[i][2])
            n = round((s[0] - self._state_descriptions[i][0]) / delta)

            val = self._state_descriptions[i][0] + n * delta
            if val > self._state_descriptions[i][1]:
                val = self._state_descriptions[i][1]
            elif val < self._state_descriptions[i][0]:
                val = self._state_descriptions[i][0]

            state_norm.append(val)

        return tuple(state_norm)
