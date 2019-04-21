from typing import List, Union

from game import ActionType, Game
from player.base import Player
from utils.consoledisplay import display_state, display_action


def play_batch(n_players: int, players: Union[Player, List[Player]], n_games: int = 2):
    if isinstance(players, Player):
        players = [players] * n_players  # same instance of player

    games = [Game(n_players) for _ in range(n_games)]

    n_actions = games[0].n_actions
    action_freq = [[0] * n_actions for _ in range(n_players)]
    # number of play made of each type
    play_known_success = 0
    play_known_fail = 0
    play_unknown_success = 0
    play_unknown_fail = 0
    play_partial_success = 0
    play_partial_fail = 0
    while not all(g.is_over for g in games):
        cur_player_id = games[0].cur_player
        cur_player = players[cur_player_id]
        actions = cur_player.get_batch_actions(games)
        for game, action_ind in zip(games, actions):
            if not game.is_over:
                action = game.actions[action_ind]
                # keep track of the type of play
                if action.type == ActionType.PLAY:
                    tile = game.hands[cur_player_id][action.target_tile]
                    hint = game.hints[cur_player_id][action.target_tile]
                    pos_suits = hint[0].count(True)
                    pos_ranks = hint[1].count(True)
                    success = tile.rank == game.fireworks[tile.suit]
                    if pos_suits > 1 and pos_ranks > 1:
                        if success:
                            play_unknown_success += 1
                        else:
                            play_known_fail += 1
                    elif pos_suits > 1 or pos_ranks > 1:
                        if success:
                            play_partial_success += 1
                        else:
                            play_partial_fail += 1
                    else:
                        if success:
                            play_known_success += 1
                        else:
                            play_known_fail += 1
                game.play(action)
                action_freq[cur_player_id][action_ind] += 1

    return \
        games, action_freq, \
        play_known_success, play_known_fail, \
        play_partial_success, play_partial_fail, \
        play_unknown_success, play_unknown_fail


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
