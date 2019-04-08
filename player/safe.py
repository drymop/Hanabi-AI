import numpy as np

from game import Game
from player.base import Player


class SafePlayer(Player):
    def get_action(self, game: Game):
        # if there is a tile that is surely playable (based solely on hint), play it
        can_play = []
        for i, tile in enumerate(game.hands[game.cur_player]):  # for each
            if game.fireworks[tile.suit] == tile.rank \
                    and game.hints[game.cur_player][i][0].count(True) == 1 \
                    and game.hints[game.cur_player][i][1].count(True) == 1:
                can_play.append(i)
        if can_play:
            return np.random.choice(can_play)

        # otherwise, if there is no hint token, discard a random tile
        if game.n_hint_tokens == 0:
            return np.random.randint(game.hand_size, game.hand_size * 2)  # discard actions are in this range

        # otherwise, hint playable tile of the next player
        # for each attribute, figure out how many sure playable tile is created
        n_hint_attributes = Game.N_RANKS + Game.N_SUITS
        hint_action_ind_start = game.hand_size * 2  # index start after play actions and discard actions
        for p in range(1, game.n_players):
            player = (p + game.cur_player) % game.n_players
            hand = game.hands[player]
            hint = game.hints[player]
            full_playable_rank = [0] * Game.N_RANKS  # number of playable tile results from hinting this rank
            full_playable_suit = [0] * Game.N_SUITS  # number of playable tile results from hinting this suit
            part_playable_rank = [0] * Game.N_RANKS  # number of playable tile results from hinting this rank
            part_playable_suit = [0] * Game.N_SUITS  # number of playable tile results from hinting this suit
            for i, tile in enumerate(hand):
                if game.fireworks[tile.suit] == tile.rank:
                    possible_ranks = hint[i][0].count(True)
                    possible_suits = hint[i][1].count(True)
                    if possible_ranks == 1 and possible_suits == 1:  # already fully playable
                        continue
                    elif possible_ranks != 1 and possible_suits != 1:  # both ranks and suits are unknown
                        part_playable_rank[tile.rank] += 1
                        part_playable_suit[tile.suit] += 1
                    elif possible_ranks != 1:  # unknown rank, known suit
                        full_playable_rank[tile.rank] += 1
                    else:  # unknown suit, known rank
                        full_playable_suit[tile.suit] += 1
            # if there are hints that make a tile fully playable, hint
            full_playable = full_playable_rank + full_playable_suit
            max_playable = max(full_playable)
            if max_playable > 0:
                choices = [i for i in range(len(full_playable)) if full_playable[i] == max_playable]
                return np.random.choice(choices) + hint_action_ind_start
            # else if there are hints that make a tile partially playable, hint
            part_playable = part_playable_rank + part_playable_suit
            max_playable = max(part_playable)
            if max_playable > 0:
                choices = [i for i in range(len(full_playable)) if part_playable[i] == max_playable]
                return np.random.choice(choices) + hint_action_ind_start

            # no playable tile in this player hand, continue to next player
            hint_action_ind_start += n_hint_attributes

        # no player has any playable tile, do a random non-play action
        choices = [i for i in range(game.hand_size, game.n_actions) if game.is_valid_action[i]]
        return np.random.choice(choices)
