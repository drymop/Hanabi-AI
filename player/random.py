import numpy as np
from typing import List

from game import Game


class RandomPlayer:
    def get_action(self, game: Game):
        choices = [i for i in range(game.n_actions) if game.is_valid_action[i]]
        return np.random.choice(choices)


class RandomOnlyPlayPlayer:
    def get_action(self, game):
        return np.random.choice(game.hand_size)
