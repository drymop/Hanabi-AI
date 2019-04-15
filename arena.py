from typing import List, Union

from game import Game
from player.base import Player
from utils.consoledisplay import display_state, display_action


def play_batch(n_players: int, players: Union[Player, List[Player]], n_games: int = 2):
    if isinstance(players, Player):
        players = [players] * n_players  # same instance of player

    games = [Game(n_players) for _ in range(n_games)]

    n_actions = games[0].n_actions
    action_freq = [[0] * n_actions for _ in range(n_players)]
    while not all(g.is_over for g in games):
        cur_player = players[games[0].cur_player]
        actions = cur_player.get_batch_actions(games)
        for game, action_ind in zip(games, actions):
            if not game.is_over:
                game.play(game.actions[action_ind])
                action_freq[game.cur_player][action_ind] += 1

    return games, action_freq

def play_verbose(n_players: int, players: Union[Player, List[Player]], wait_each_turn: bool = False):
    if isinstance(players, Player):
        players = [players] * n_players  # same instance of player
    game = Game(n_players)

    while not game.is_over:
        display_state(game, first_person=False)

        cur_player = players[game.cur_player]
        action_ind = cur_player.get_action(game, verbose=True)
        action = game.actions[action_ind]
        print('Action ind: %d' % action_ind)
        display_action(game, action)
        print()
        game.play(action)
        if wait_each_turn:
            input()
    display_state(game, first_person=False)
    if wait_each_turn:
        input()