from typing import List

from game import Game


class Player:
    def get_action(self, game: Game, verbose: bool = False) -> int:
        raise NotImplementedError()

    def get_batch_actions(self, games: List[Game]) -> List[int]:
        return [self.get_action(game) for game in games]
