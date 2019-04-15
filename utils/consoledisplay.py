from typing import List

from game import ActionType
from game import Game

RANKS = ('1', '2', '3', '4', '5')
COLORS = ('B', 'G', 'R', 'W', 'Y')


def display_state(game: Game, first_person=False):
    n = game.n_players
    s = game.hand_size

    print('-' * 120)

    gap_len = 7
    hand_len = s * 7 - 2

    sep = '-' * hand_len
    gap = ' ' * gap_len
    sep_row = gap.join([sep] * n)

    tile_format = ' %s%s  '
    tile_row = '  '.join([tile_format] * s)
    tile_row = gap.join([tile_row] * n)

    # print hints and fuses
    print('Player   : %d \nHints    : %d \nFuses    : %d\n' %
          (game.cur_player + 1, game.n_hint_tokens, game.n_fuse_tokens))

    # print firework pile
    pile_str = '  '.join(['%d%s'] * Game.N_SUITS)
    pile = sum(((game.fireworks[col], COLORS[col]) for col in range(Game.N_SUITS)), ())
    print('Firework : ', end="")
    print(pile_str % pile)
    print()

    # print the tiles players' hands
    hands = sum(([RANKS[t.rank], COLORS[t.suit]] for p in range(0, n) for t in game.hands[p]), [])
    if first_person:
        two_s = 2 * s
        hands[:] = [hands[i] if (i // two_s != game.cur_player) else '?' for i in range(0, len(hands))]
    hands = tuple(hands)

    print(sep_row)
    print(tile_row % hands)
    print(sep_row)

    # print the hints
    hint_str = '  '.join(['%s' * game.N_SUITS] * s)
    hint_str = gap.join([hint_str] * n)
    hints = [RANKS[i] if game.hints[p][t][0][i] else '_' for p in range(0, n) for t in range(0, s) for i in
             range(0, game.N_RANKS)]
    print(hint_str % tuple(hints))
    hints = [COLORS[i] if game.hints[p][t][1][i] else '_' for p in range(0, n) for t in range(0, s) for i in
             range(0, game.N_SUITS)]
    print(hint_str % tuple(hints))
    print("End")


def display_action(game: Game, action):
    print('Action: ', end="")
    if action.type == ActionType.PLAY:
        tile = game.hands[game.cur_player][action.target_tile]
        print('Play tile {} ({}{})'.format(action.target_tile + 1, RANKS[tile.rank], COLORS[tile.suit]))
    elif action.type == ActionType.DISCARD:
        tile = game.hands[game.cur_player][action.target_tile]
        print('Discard tile {} ({}{})'.format(action.target_tile + 1, RANKS[tile.rank], COLORS[tile.suit]))
    elif action.type == ActionType.HINT:
        attr = COLORS[action.hint_attribute] if action.hint_is_suit else RANKS[action.hint_attribute]
        target = (game.cur_player + action.target_player) % game.n_players + 1
        print('Hint player {} of all {} tiles'.format(target, attr))
    else:
        print('Unknown action')

def display_action_distribution(distr: List[float], col_width: int = 5, n_decimals: int = 1):
    ind_format = ('%' + str(col_width) + 'd') * len(distr)
    print(ind_format % tuple(i for i in range(len(distr))))
    format = ('%' + str(col_width) + '.' + str(n_decimals) + 'f') * len(distr)
    print(format % tuple(distr))
