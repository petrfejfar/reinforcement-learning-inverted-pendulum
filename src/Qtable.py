from random import getrandbits


class QTable:
    def __init__(self, actions, states):
        """
            actions [-10, 10]
            states [(min, max, inteval_counts),...]
        """
        self._gamma = 0.9
        self._actions = list(range(len(actions)))
        self._action_map = actions

    def get_best_action(self, s):
        # TODO
        return self._action_map[getrandbits(1)]

    def get_random_action(self, s):
        pass

    def update_reward(self, s, s1, r):
        pass

    def _normalize_state(self, s):
        pass
