import numpy as np
from typing import List

from game import Game


class RandomPlayer:
    def __init__(self, n_players, prob: List[float] = None):
        self.n_actions = Game.ACTIONS_PER_N_PLAYERS[n_players] - 1  # exclude doing nothing
        self.prob = prob

    def get_action(self):
        return np.random.choice(self.n_actions, p=self.prob)


class RandomNoHintPlayer:
    def __init__(self, n_players):
        self.n_actions = Game.HAND_SIZE_PER_N_PLAYERS[n_players]

    def get_action(self):
        return np.random.choice(self.n_actions)
