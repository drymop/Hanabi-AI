from collections import namedtuple
from copy import deepcopy
from itertools import product as crossproduct
from random import randint, shuffle
import numpy as np

Tile = namedtuple('Tile', 'rank color id')
GameDelta = type('GameDelta', (object,), {})

class Game(object):
    """
    Object represents a standard Game of Hanabi. Containing the following game state information:
    deck: the undrawn tiles

    """

    RANKS    = (0, 1, 2, 3, 4)
    COLORS   = (0, 1, 2, 3, 4)
    N_RANKS  = len(RANKS)
    N_COLORS = len(COLORS)
    # list of all unique tile types (= RANKS x COLORS)
    # to translate between tile type and tile index, the formulat is
    # tileId = rank * 5 + color
    TILE_TYPES = tuple(Tile(t[0], t[1], 5 * t[0] + t[1]) for t in crossproduct(RANKS, COLORS))
    N_TYPES    = len(TILE_TYPES)
    # Number of tile for each rank of a color. E.g there are only 1 Five of each color, 
    # so Game.N_TILES_PER_RANK[4] = 1.
    N_TILES_PER_RANK = (3, 2, 2, 2, 1)
    # e.g if there are 4 players, hand size = HAND_SIZE_PER_N_PLAYERS[4] = 4
    HAND_SIZE_PER_N_PLAYERS = (None, None, 5, 5, 4, 4)

    # Hint mask
    HINT_ALL_RANKS  = ((1 << N_RANKS) - 1) << N_COLORS
    HINT_ALL_COLORS = (1 << N_COLORS) - 1
    HINT_ALL        = HINT_ALL_RANKS | HINT_ALL_COLORS
    HINT_COLORS     = tuple(1 << i for i in range(0, N_COLORS))
    HINT_RANKS      = tuple(1 << i for i in range(N_COLORS, N_COLORS + N_RANKS))


    def __init__(self, nPlayers):
        # keep count of each tile type
        self.nTypeInPlay = [Game.N_TILES_PER_RANK[t.rank] for t in Game.TILE_TYPES]

        # initial deck, containing all tiles
        # each tile is represented by a number, which is the index of the type in Game.TILE_TYPES list
        # as tiles are drawn, deck get shuffled and partitioned into undrawn pile (index 0 to deckSize-1)
        # and drawn pile (index deckSize or greater)
        self.deck = [t for t in Game.TILE_TYPES for j in range(0, Game.N_TILES_PER_RANK[t.rank]) ]
        shuffle(self.deck)
        # if dterminized, the draw sequence is fixed
        self.determinized = False
        # remaining tiles in deck
        self.deckSize = len(self.deck)

        # winning pile
        self.fireworkPile = [0] * Game.N_COLORS

        # game tokens
        self.nHintTokens = 8
        self.nFuseTokens = 3

        # game score
        self.score = 0
        self.MAX_SCORE = Game.N_COLORS * Game.N_RANKS

        # number of players
        self.curPlayer  = 0
        self.NUM_PLAYERS = nPlayers

        # players hand
        self.HAND_SIZE   = Game.HAND_SIZE_PER_N_PLAYERS[self.NUM_PLAYERS]
        self.hands = [ [self._drawTile() for i in range(0, self.HAND_SIZE)] for p in range(0, self.NUM_PLAYERS) ]

        # players hint, true if an attribute (rank or color) is possible for a tile
        # the attribute in order are [C0, C1, C2, C3, C4, V1, V2, V3, V4, V5]
        self.hints = [[Game.HINT_ALL] * self.HAND_SIZE for p in range(0, self.NUM_PLAYERS)]

        # game history
        self.history = []


    def isOver(self):
        """
        Returns: True if game is over
        """
        return self.nFuseTokens <= 0 or self.deckSize <= 0 or self.score == 25


    def play(self, action):
        if action.type == 'PLAY':
            self._processPlay(action)
        elif action.type == 'DISCARD':
            self._processDiscard(action)
        elif action.type == 'HINT':
            self._processHint(action)
        else:
            raise ValueError("Expected action type to be 'PLAY', 'DISCARD' or 'HINT'; got '%r'".format(action.type))
        
        # move to next player
        self.curPlayer = (self.curPlayer + 1) % self.NUM_PLAYERS


    def undo(self):
        delta = self.history.pop()
        # restore player
        self.curPlayer = (self.curPlayer - 1) % self.NUM_PLAYERS

        if delta.actionType == 'PLAY':
            # return the played tile with hint to hand 
            # and add it to number of tiles in play
            self.hands[self.curPlayer][delta.playedPos] = delta.playedTile
            self.hints[self.curPlayer][delta.playedPos] = delta.oldHint
            self.nTypeInPlay[delta.playedTile.id] += 1
            # return the drawn tile to the deck
            self.deckSize += 1

            if delta.playCorrect:
                self.fireworkPile[delta.playedTile.color] -= 1
                self.score -= 1
                if (delta.playedTile.rank == self.N_RANKS - 1):
                    # undo a firework completion
                    self.nHintTokens -= 1
            else:
                self.nFuseTokens += 1
        elif delta.actionType == 'DISCARD':
            # return the discarded tile with hint to hand 
            # and add it to number of tiles in play
            self.hands[self.curPlayer][delta.discardPos] = delta.discardedTile
            self.hints[self.curPlayer][delta.discardPos] = delta.oldHint
            self.nTypeInPlay[delta.discardedTile.id] += 1
            # return the drawn tile to the deck
            self.deckSize += 1
            # remove the gained hint from discard
            self.nHintTokens -= 1
        else: # give hint
            # return the hint token
            self.nHintTokens += 1
            self.hints[delta.hintedPlayer] = delta.oldHints


    def _processPlay(self, action):
        # changes made in this turn
        delta = GameDelta()
        delta.actionType = action.type
        delta.playedPos  = action.targetTile
        delta.oldHint    = self.hints[self.curPlayer][action.targetTile]

        tile = self.hands[self.curPlayer][action.targetTile]
        delta.playedTile = tile
        # remove the played tile from play
        self.nTypeInPlay[tile.id] -= 1
        # draw new tile, init hint for the new tile
        self.hands[self.curPlayer][action.targetTile] = self._drawTile()    
        self.hints[self.curPlayer][action.targetTile] = Game.HINT_ALL

        if tile.rank == self.fireworkPile[tile.color]:
            # correct tile placement: add to firework pile
            self.fireworkPile[tile.color] += 1
            self.score += 1
            if self.fireworkPile[tile.color] == Game.N_RANKS:
                # if completing full pile of a color, add a hint token
                self.nHintTokens += 1
            
            delta.playCorrect = True
        else:
            # incorrect tile placement: remove a fuse
            self.nFuseTokens -= 1
            delta.playCorrect = False

        self.history.append(delta)


    def _processDiscard(self, action):
        # changes made in this turn
        delta = GameDelta()
        delta.actionType = action.type


        tile = self.hands[self.curPlayer][action.targetTile]
        # save info about the tile discarded and its hint
        delta.discardPos = action.targetTile
        delta.discardedTile = tile
        delta.oldHint = self.hints[self.curPlayer][action.targetTile]

        # dicard the tile
        self.nTypeInPlay[tile.id] -= 1
        # draw new tile and init its hint
        self.hands[self.curPlayer][action.targetTile] = self._drawTile()
        self.hints[self.curPlayer][action.targetTile] = Game.HINT_ALL
        # gain a new hint
        self.nHintTokens += 1


    def _processHint(self, action):
        # changes made in this turn
        delta = GameDelta()
        delta.actionType = action.type

        # remove a hint token
        self.nHintTokens -= 1
        # get target player and attribute
        player = self.curPlayer + action.targetPlayer
        attribute = action.hintAttribute
        delta.hintedPlayer = player
        delta.oldHints = deepcopy(self.hints[player])

        if (action.hintIsColor):
            for i in range(0, self.HAND_SIZE):
                if self.hands[player][i].color == attribute:
                    self.hints[player][i] &= Game.HINT_ALL_RANKS | Game.HINT_COLORS[attribute]
                else:
                    self.hints[player][i] &= ~Game.HINT_COLORS[attribute]
        else:
            for i in range(0, self.HAND_SIZE):
                if self.hands[player][i].rank == attribute:
                    self.hints[player][i] &= Game.HINT_ALL_COLORS | Game.HINT_RANKS[attribute]
                else:
                    self.hints[player][i] &= ~Game.HINT_RANKS[attribute]


    def _drawTile(self):
        """
        Choose a random tile from undrawn part of deck, then switch it to the start of the drawn part
        and return that tile index.
        If deck is empty, return None
        """
        if self.deckSize == 0:
            return None
        self.deckSize -= 1

        # if random drawing, choose a random tile and swap it to the end of the
        # deck, where the deck meet the drawn part
        # in both cases, return the tile at the end of the deck
        if not self.determinized:
            # choose a random tile in deck
            r = randint(0, self.deckSize)
            # move the drawn tile to the drawn part
            self.deck[r], self.deck[self.deckSize] = self.deck[self.deckSize], self.deck[r]
        
        # return the drawn tile
        return self.deck[self.deckSize]  