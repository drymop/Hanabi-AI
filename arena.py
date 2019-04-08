from typing import List, Union

from game import Game
from player.base import Player


def play_batch(n_players: int, players: Union[Player, List[Player]], n_games: int = 2):
    if isinstance(players, Player):
        players = [players] * n_players  # same instance of player

    games = [Game(n_players) for _ in range(n_games)]

    while not all(g.is_over for g in games):
        cur_player = players[games[0].cur_player]
        actions = cur_player.get_batch_actions(games)
        for game, action_ind in zip(games, actions):
            if not game.is_over:
                game.play(game.actions[action_ind])

    return games
