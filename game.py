from collections import namedtuple
from enum import Enum
from itertools import product as cartesian_product
from random import shuffle


Tile = namedtuple('Tile', 'rank suit id')


class ActionType(Enum):
    PLAY = 0
    HINT = 1
    DISCARD = 2
    NONE = 3


class Action(object):
    def __init__(self, action_type):
        self.type = action_type


class Game(object):
    """Object represents a standard Game of Hanabi."""

    RANKS = (0, 1, 2, 3, 4)
    SUITS = (0, 1, 2, 3, 4)
    N_RANKS = len(RANKS)
    N_SUITS = len(SUITS)
    MAX_SCORE = N_SUITS * N_RANKS
    MAX_HINTS = 8
    MAX_FUSES = 10

    # list of all unique tile types (= RANKS x SUITS)
    # to translate between tile type and tile index, the formula is
    # tileId = rank * 5 + color
    TILE_TYPES = tuple(Tile(rank=t[0], suit=t[1], id=5*t[0]+t[1])for t in cartesian_product(RANKS, SUITS))
    N_TYPES = len(TILE_TYPES)
    # Number of tile for each rank of a color. E.g there are only 1 Five of each color, 
    # so Game.N_TILES_PER_RANK[4] = 1.
    N_TILES_PER_RANK = (3, 2, 2, 2, 1)
    # e.g if there are 4 players, hand size = HAND_SIZE_PER_N_PLAYERS[4] = 4
    HAND_SIZE_PER_N_PLAYERS = (None, None, 5, 5, 4, 4)
    ACTIONS_PER_N_PLAYERS = (None, None, None, 31, None, None)

    def __init__(self, n_players):
        # keep count of each tile type
        self.n_tiles_per_type = [Game.N_TILES_PER_RANK[t.rank] for t in Game.TILE_TYPES]

        # initial deck, containing all tiles
        # each tile is represented by a number, which is the index of the type in Game.TILE_TYPES list
        # as tiles are drawn, deck get shuffled and partitioned into undrawn pile (index 0 to deck_size-1)
        # and drawn pile (index deck_size or greater)
        self.deck = [t for t in Game.TILE_TYPES for _ in range(Game.N_TILES_PER_RANK[t.rank])]
        shuffle(self.deck)
        # remaining tiles in deck
        self.deck_size = len(self.deck)

        # winning pile
        self.fireworks = [0] * Game.N_SUITS

        # game tokens
        self.n_hint_tokens = Game.MAX_HINTS
        self.n_fuse_tokens = Game.MAX_FUSES

        # game progress
        self.score = 0
        self.n_turns = 0

        # number of players
        self.cur_player = 0
        self.n_players = n_players

        # players hand
        self.hand_size = Game.HAND_SIZE_PER_N_PLAYERS[self.n_players]
        # self.hands[player_index][tile_index]
        self.hands = [[self._draw_tile() for _ in range(self.hand_size)] for __ in range(self.n_players)]

        # players hint, true if an attribute (rank or color) is possible for a tile
        # the attribute in order are [S0, S1, S2, S3, S4, R0, R1, R2, R3, R4]
        self.hints = [[[[True] * Game.N_RANKS, [True] * Game.N_SUITS]
                       for _ in range(self.hand_size)] for __ in range(self.n_players)]

        # actions players can take
        self.actions = []
        for i in range(self.hand_size):
            ac = Action(ActionType.PLAY)
            ac.target_tile = i
            self.actions.append(ac)
        for i in range(self.hand_size):
            ac = Action(ActionType.DISCARD)
            ac.target_tile = i
            self.actions.append(ac)
        for p in range(1, self.n_players):
            for r in Game.RANKS:
                ac = Action(ActionType.HINT)
                ac.target_player = p
                ac.hint_attribute = r
                ac.hint_is_suit = False
                self.actions.append(ac)
            for c in Game.SUITS:
                ac = Action(ActionType.HINT)
                ac.target_player = p
                ac.hint_attribute = c
                ac.hint_is_suit = True
                self.actions.append(ac)
        self.actions.append(Action(ActionType.NONE))  # do nothing
        self.n_actions = Game.ACTIONS_PER_N_PLAYERS[self.n_players]

        # whether game is over or not
        self.is_over = False

        # list of valid actions for current player
        self.is_valid_action = [True] * self.n_actions
        self.is_valid_action[-1] = False  # current player cannot do nothing during his turn
        self._recalculate_valid_actions()

    def _recalculate_valid_actions(self):
        """Recalculates which actions are valid, stores the result in self.is_valid_action"""
        # if game is over, no action is possible
        if self.is_over:
            for i in range(self.n_actions):
                self.is_valid_action[i] = False
                return

        act_ind = 2 * self.hand_size  # skip play and discard actions, as they are always valid
        # cannot hint
        if self.n_hint_tokens <= 0:
            for i in range(act_ind, self.n_actions-1):
                self.is_valid_action[i] = False
            return

        # hint to each player
        for i in range(1, self.n_players):
            p = (self.cur_player + i) % self.n_players
            hand = self.hands[p]
            has_rank = [False] * Game.N_RANKS
            has_suit = [False] * Game.N_SUITS
            for tile in hand:
                has_rank[tile.rank] = True
                has_suit[tile.suit] = True
            self.is_valid_action[act_ind:act_ind + Game.N_RANKS] = has_rank
            act_ind += Game.N_RANKS
            self.is_valid_action[act_ind:act_ind + Game.N_SUITS] = has_suit
            act_ind += Game.N_SUITS

    def play(self, action):
        if action.type == ActionType.PLAY:
            self._process_play(action)
        elif action.type == ActionType.DISCARD:
            self._process_discard(action)
        elif action.type == ActionType.HINT:
            self._process_hint(action)
        else:
            raise ValueError("Expected action type to be 'PLAY', 'DISCARD' or 'HINT'; got '%r'" % action.type)
        
        # move to next player
        self.n_turns += 1
        self.cur_player = (self.cur_player + 1) % self.n_players
        self.is_over = self.n_fuse_tokens <= 0 or self.deck_size <= 0 or self.score == Game.MAX_SCORE
        self._recalculate_valid_actions()

    def _process_play(self, action):
        tile = self.hands[self.cur_player][action.target_tile]
        # remove the played tile from play
        self.n_tiles_per_type[tile.id] -= 1
        # draw new tile, init hint for the new tile
        self.hands[self.cur_player][action.target_tile] = self._draw_tile()
        self.hints[self.cur_player][action.target_tile] = [[True] * Game.N_RANKS, [True] * Game.N_SUITS]

        if tile.rank == self.fireworks[tile.suit]:
            # correct tile placement: add to firework pile
            self.fireworks[tile.suit] += 1
            self.score += 1
            if self.fireworks[tile.suit] == Game.N_RANKS:
                # if completing full pile of a color, add a hint token
                self.n_hint_tokens = min(Game.MAX_HINTS, self.n_hint_tokens + 1)
        else:
            # incorrect tile placement: remove a fuse
            self.n_fuse_tokens -= 1

    def _process_discard(self, action):
        tile = self.hands[self.cur_player][action.target_tile]
        # discard the tile
        self.n_tiles_per_type[tile.id] -= 1
        # draw new tile and init its hint
        self.hands[self.cur_player][action.target_tile] = self._draw_tile()
        self.hints[self.cur_player][action.target_tile] = [[True] * Game.N_RANKS, [True] * Game.N_SUITS]
        # gain a new hint
        self.n_hint_tokens = min(Game.MAX_HINTS, self.n_hint_tokens + 1)

    def _process_hint(self, action):
        # remove a hint token
        self.n_hint_tokens -= 1
        # get target player and attribute
        player = (self.cur_player + action.target_player) % self.n_players
        attribute = action.hint_attribute

        if action.hint_is_suit:
            for i in range(self.hand_size):
                match = self.hands[player][i].suit == attribute
                tile_hint = self.hints[player][i][1]
                for j in range(Game.N_SUITS):
                    if j == attribute:
                        tile_hint[j] = match
                    else:
                        tile_hint[j] = tile_hint[j] and not match
        else:
            for i in range(self.hand_size):
                match = self.hands[player][i].rank == attribute
                tile_hint = self.hints[player][i][0]
                for j in range(Game.N_RANKS):
                    if j == attribute:
                        tile_hint[j] = match
                    else:
                        tile_hint[j] = tile_hint[j] and not match

    def _draw_tile(self):
        """
        Choose a random tile from undrawn part of deck, then switch it to the start of the drawn part
        and return that tile index.
        If deck is empty, return None
        """
        if self.deck_size == 0:
            return None
        self.deck_size -= 1
        return self.deck[self.deck_size]
