import numpy as np

from game import Game
from player.base import Player


class RandomPlayer(Player):
    def get_action(self, game: Game) -> int:
        choices = [i for i in range(game.n_actions) if game.is_valid_action[i]]
        if not choices:
            return -1
        return np.random.choice(choices)


class RandomOnlyPlayPlayer(Player):
    def get_action(self, game: Game) -> int:
        return np.random.choice(game.hand_size)
