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

from math import cos, sin, floor, pi, e

class QTable:
    iterations = 1500
    max_state_transitions =  30000

    pit = ("-100","-100","-100","-100")

    def __init__(self, actions, states):
        """
            actions [-10, 10]
            states [(min, max, inteval_counts),...]
        """
        self._learning_rate = 0.1
        self._gamma = 1
        # self._learning_rate = 0.2
        # self._gamma = 0.9

        self._actions = list(range(len(actions)))
        print (self._actions)
        self._action_map = actions
        self._state_descriptions = states

        state_len = len(self._state_descriptions)
        values = []
        # state_cnt = 0

        for i in range(state_len):
            delta = (self._state_descriptions[i][1] - self._state_descriptions[i][0]) / float(self._state_descriptions[i][2])
            values.append([])
            for x in range(self._state_descriptions[i][2] + 1):
                val = format(self._state_descriptions[i][0] + x * delta, '.6f')
                values[i].append(val)
            print(values[i])
            # state_cnt = len(values[i])

        # self._

        all_states = list(itertools.product(*values))
        all_states.append(("-100","-100","-100","-100"))
        # val = [values[0], values[1]]
        # all_states = list(itertools.product(*val))
        # print("")
        # print("")
        # print("")
        # print(all_states)

        self._table = dict((el, [0] * len(actions)) for el in all_states)

    def get_q_vals(self, s):
        state_norm = self._normalize_state(s)
        return self._table[state_norm]

    def _get_best_action_index(self, s):
        state_norm = self._normalize_state(s)
        return self._table[state_norm].index(max(self._table[state_norm]))

    def get_best_action(self, s):
        return self._action_map[self._get_best_action_index(s)]

    def learn(self, simulate_f, reward_f, is_safe):

        for i in range(1, self.iterations+1):
            state = (0.0, 0.0, 0.0, 0.0)
            state_norm = self._normalize_state(state)

            transition = 0
            success = False

            updates = []

            while True:
                transition += 1

                a = self._get_random_action(state)
                # a = self._get_best_action_index(state)
                # r = reward_f(state)

                next_s = simulate_f(self._action_map[a], state)
                r = reward_f(next_s)
                next_s_norm = self._normalize_state(next_s)

                updates.append((state_norm, a, next_s_norm, r))
                # self._update_Q(state_norm, a, next_s_norm, r)

                if not is_safe(next_s):
                    break

                if transition >= self.max_state_transitions:
                    success = True
                    break


                state = next_s
                state_norm = self._normalize_state(state)

            print(updates[-1])
            for state_norm, a, next_s_norm, r in reversed(updates):
                self._update_Q(state_norm, a, next_s_norm, r)

            # if not is_safe(state):
            if not is_safe(next_s):
                # print ("\nAlmost last state: ", state, self._table[state_norm])
                print ("Last state: ", state, self._table[state_norm])

            print("Iteration #%05d %s after %d steps." % (i, "success" if success else "failed", transition))
            if success:
                break

            # print("Q table")
            # for x in self._table:
            #     print(x, self._table[x][0], self._table[x][1])

        # for i in range(1, self.iterations+1):
        #     if i % 100 is 0:
        #         print("Iteration #%05d" % i)
        #
        #     state = (0.0, 0.0, 0.0, 0.0)
        #     state_norm = self._normalize_state(state)
        #
        #     for x in range(self.state_transitions):
        #         a = self._get_random_action(state_norm)
        #         r = reward_f(state)
        #
        #         next_s = simulate_f(self._action_map[a], state)
        #         next_s_norm = self._normalize_state(next_s)
        #
        #         self._update_Q(state_norm, a, next_s_norm, r)
        #
        #         if not is_safe(state):
        #             break
        #
        #         state = next_s
        #         state_norm = self._normalize_state(state)

    def _get_random_action(self, s):
        state_norm = self._normalize_state(s)

        T = 4
        kappa = e
        total = 0

        max_w = max(self._table[state_norm])
        for w in self._table[state_norm]:
            total += kappa**((w-max_w)/T)

        r = random.random()
        upto = 0
        choice = 0

        for c, w in enumerate(self._table[state_norm]):
            delta_w = kappa**((w-max_w)/T) / total

            if upto + delta_w >= r:
                choice = c
                break
            upto += delta_w

        return self._actions[choice]

    def _update_Q(self, s, a, s1, r):
        # print(self._table[s1])
        # print(max(self._table[s1]))

        # print(s1)

        sample = r + self._gamma * max(self._table[s1])

        # if s1 == self.pit:
        #     print("%f + %f * %f" % (r, self._gamma, max(self._table[s1])))
        #     print("%f + %f * (%f - %f)" % (self._table[s][a], self._learning_rate, sample, self._table[s][a]))

        self._table[s][a] = self._table[s][a] + self._learning_rate * (sample - self._table[s][a])

        # print("Old val ", self._table[s][a])
        # print("reward ", r)
        # self._table[s][a] = r + self._gamma * max(self._table[s1])
        # print("New val ", self._table[s][a])

        # self._table[s][a] = self._table[s][a] + self._learning_rate * (r + self._gamma * max(self._table[s1]))

    def _normalize_state(self, s):
        if not self.system_safe2(s):
            return ("-100","-100","-100","-100")

        state_len = len(self._state_descriptions)

        state_norm = []
        for i in range(state_len):
            delta = (self._state_descriptions[i][1] - self._state_descriptions[i][0]) / float(self._state_descriptions[i][2])
            n = round((s[i] - self._state_descriptions[i][0]) / delta)

            val = self._state_descriptions[i][0] + n * delta

            if val > self._state_descriptions[i][1]:
                val = self._state_descriptions[i][1]
            elif val < self._state_descriptions[i][0]:
                val = self._state_descriptions[i][0]

            val = format(val, '.6f')
            state_norm.append(val)

        return tuple(state_norm)

    def system_safe2(self, state):
        """Is the system in stable?"""
        theta = state[2] - (2*pi) * floor((state[2] + pi) / (2*pi))
        return abs(state[0]) < 2.4 and abs(theta) < 0.20943951
