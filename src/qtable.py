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

from math import e
import numpy as np
import matplotlib.pyplot as plt

class QTable:
    """Represent and learn Q table"""

    pit = ("-100", "-100", "-100", "-100")                      # terminal state

    def __init__(self, actions, states):
        """
            actions [-10, 10]
            states [(min, max, inteval_counts),...]
        """

        # Assign dummy values
        self._model = None
        self._learning_rate = 0
        self._gamma = 0

        # Save data
        self._actions = list(range(len(actions)))
        self._action_map = actions
        self._state_descriptions = states

        state_len = len(self._state_descriptions)
        values = []

        # Discretize state coordinates
        for i in range(state_len):
            # Coordinate description
            val_min = self._state_descriptions[i][0]
            val_max = self._state_descriptions[i][1]
            val_pieces = self._state_descriptions[i][2]

            delta = (val_max - val_min) / float(val_pieces)

            values.append([])

            # Generate all possible discrete values
            for x in range(self._state_descriptions[i][2] + 1):
                val = format(self._state_descriptions[i][0] + x * delta, '.6f')
                values[i].append(val)

        # Generate all possible states
        all_states = list(itertools.product(*values))
        all_states.append(self.pit)                         # add terminal state

        # Initialize QTable
        self._table = dict((el, [0] * len(actions)) for el in all_states)

    def get_q_vals(self, s):
        """Return Q values for all actions"""

        state_norm = self._normalize_state(s)
        return self._table[state_norm]

    def _get_best_action_index(self, s):
        """Return best action index based on Q value"""

        state_norm = self._normalize_state(s)
        return self._table[state_norm].index(max(self._table[state_norm]))

    def get_best_action(self, s):
        """Return best action"""

        return self._action_map[self._get_best_action_index(s)]

    def learn(self, model, iterations, max_state_transitions, \
              simulation_timespan, learning_rate=0.1, discount_factor=1):
        """Learn Q table"""

        self._model = model
        self._learning_rate = learning_rate
        self._gamma = discount_factor

        success = False

        # For each iterarion
        for i in range(1, iterations+1):
            if success:
                break

            model.reset()
            state_norm = self._normalize_state(model.get_state())

            transition = 0

            updates = []

            # Change states forever
            while not success:
                transition += 1

                a = self._get_random_action(model.get_state())
                model.simulate(self._action_map[a], simulation_timespan)

                r = model.reward()
                next_s_norm = self._normalize_state(model.get_state())

                updates.append((state_norm, a, next_s_norm, r))

                # Stop on failure
                if not model.is_system_safe():
                    break

                # Stop on success
                if transition >= max_state_transitions:
                    success = True
                    break

                state_norm = self._normalize_state(model.get_state())

            for state_norm, a, next_s_norm, r in reversed(updates):
                self._update_q(state_norm, a, next_s_norm, r)

            print("Iteration #%05d %s after %d steps. \t" % \
                    (i, "success" if success else "failed", transition), end="")

            if not success:
                print(updates[-1], end="")

            print("")

    def _get_random_action(self, s):
        """Boltzmann random action selection"""

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

    def _update_q(self, s, a, s1, r):
        """Update Q value"""

        sample = r + self._gamma * max(self._table[s1])
        self._table[s][a] += self._learning_rate * (sample - self._table[s][a])

    def _normalize_state(self, s):
        """Normalize state"""

        if not self._model.is_state_safe(s):
            return ("-100", "-100", "-100", "-100")

        state_len = len(self._state_descriptions)
        state_norm = []

        for i in range(state_len):
            val_min = self._state_descriptions[i][0]
            val_max = self._state_descriptions[i][1]
            val_pieces = self._state_descriptions[i][2]

            delta = (val_max - val_min) / float(val_pieces)
            n = round((s[i] - val_min) / delta)

            val = val_min + n * delta

            if val > val_max:
                val = val_max
            elif val < val_min:
                val = val_min

            val = format(val, '.6f')
            state_norm.append(val)

        return tuple(state_norm)

    def get_tableindexes(self, s):
        """Return table indexes"""

        result = [0, 0, 0, 0]

        for i in range(len(s)):
            val_min = self._state_descriptions[i][0]
            val_max = self._state_descriptions[i][1]
            val_pieces = self._state_descriptions[i][2]

            delta = (val_max - val_min) / float(val_pieces)
            value = float(s[i])
            value = min(value, val_max)
            n = round((value - val_min) / delta)
            result[i] = n

        return tuple(result)

    def set_axis(self, ax, x_index, y_index):
        """Set axis"""

        # X axix
        x_val_min = self._state_descriptions[x_index][0]
        x_val_max = self._state_descriptions[x_index][1]
        x_val_pieces = self._state_descriptions[x_index][2]

        delta = (x_val_max - x_val_min) / float(x_val_pieces)
        ticks = [x_val_min]

        for i in range(x_val_pieces + 1):
            ticks.append(x_val_min + i * delta)

        ax.set_xticks(ticks)

        # Y axix
        y_val_min = self._state_descriptions[y_index][0]
        y_val_max = self._state_descriptions[y_index][1]
        y_val_pieces = self._state_descriptions[y_index][2]

        delta = (y_val_max - y_val_min) / float(y_val_pieces)
        ticks = [y_val_min]

        for i in range(y_val_pieces + 1):
            ticks.append(y_val_min + i * delta)

        ax.set_yticks(ticks)

        ax.grid(which='major', alpha=0.5)

    def draw(self, filename, simple=False):
        """Draw Q table"""

        x_index = 0
        y_index = 2

        x_sub_index = 1
        y_sub_index = 3

        if simple:
            x_sub_size = 1
            y_sub_size = 1
        else:
            y_sub_size = self._state_descriptions[y_sub_index][2] + 1
            x_sub_size = self._state_descriptions[x_sub_index][2] + 1


        x_val_min = self._state_descriptions[x_index][0]
        x_val_max = self._state_descriptions[x_index][1]
        x_val_pieces = self._state_descriptions[x_index][2]

        y_val_min = self._state_descriptions[y_index][0]
        y_val_max = self._state_descriptions[y_index][1]
        y_val_pieces = self._state_descriptions[y_index][2]


        x_size = (x_val_pieces + 1) * x_sub_size
        y_size = (y_val_pieces + 1) * y_sub_size
        data = np.zeros((x_size, y_size))

        for key, value in self._table.items():
            i = self.get_tableindexes(key)

            if any(x < 0 for x in i):
                print("skipping: ", i)
                continue

            if simple:
                x = i[x_index]
                y = i[y_index]
            else:
                x = i[x_sub_index] + (i[x_index] * x_sub_size)
                y = i[y_sub_index] + (i[y_index] * y_sub_size)
            data[x, y] = (value[0] - value[1])

        plt.clf()
        ax = plt.figure().add_subplot(1, 1, 1)
        self.set_axis(ax, x_index, y_index)
        data = data.transpose()

        delta_x = (x_val_max - x_val_min) / float(x_val_pieces)
        delta_y = (y_val_max - y_val_min) / float(y_val_pieces)
        plt.imshow(data, \
                   extent=( \
                    x_val_min, \
                    x_val_max + delta_x, \
                    y_val_min, \
                    y_val_max + delta_y \
                   ), \
                   interpolation="nearest", \
                   aspect="auto")
        plt.xlabel("x - " + str(x_index) + "x_sub" + str(x_sub_index))
        plt.ylabel("y - " + str(y_index) + "y_sub" + str(y_sub_index))
        plt.colorbar()

        plt.savefig(filename, dpi=4*96)
