from collections import namedtuple
from copy import deepcopy
from enum import Enum
from itertools import product as crossproduct
from random import randint, shuffle
import numpy as np

Tile = namedtuple('Tile', 'rank color id')

class ActionType(Enum):
    PLAY = 0
    HINT = 1
    DISCARD = 2
    NONE = 3

class Action(object):
    def __init__(self, actionType):
        self.type = actionType

class Game(object):
    """
    Object represents a standard Game of Hanabi.

    """

    RANKS    = (0, 1, 2, 3, 4)
    COLORS   = (0, 1, 2, 3, 4)
    N_RANKS  = len(RANKS)
    N_COLORS = len(COLORS)
    MAX_SCORE = N_COLORS * N_RANKS
    MAX_HINTS = 8
    MAX_FUSES = 50

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
    ACTIONS_PER_N_PLAYERS = (None, None, -1, 31, -1, -1)

    def __init__(self, nPlayers):
        # keep count of each tile type
        self.nRemainTiles = [Game.N_TILES_PER_RANK[t.rank] for t in Game.TILE_TYPES]

        # initial deck, containing all tiles
        # each tile is represented by a number, which is the index of the type in Game.TILE_TYPES list
        # as tiles are drawn, deck get shuffled and partitioned into undrawn pile (index 0 to deckSize-1)
        # and drawn pile (index deckSize or greater)
        self.deck = [t for t in Game.TILE_TYPES for j in range(Game.N_TILES_PER_RANK[t.rank]) ]
        shuffle(self.deck)
        # if dterminized, the draw sequence is fixed
#        self.determinized = False
        # remaining tiles in deck
        self.deckSize = len(self.deck)

        # winning pile
        self.fireworks = [0] * Game.N_COLORS

        # game tokens
        self.nHintTokens = Game.MAX_HINTS
        self.nFuseTokens = Game.MAX_FUSES

        # game score
        self.score = 0
        self.MAX_SCORE = Game.N_COLORS * Game.N_RANKS

        # number of players
        self.curPlayer  = 0
        self.nPlayers = nPlayers

        # players hand
        self.handSize   = Game.HAND_SIZE_PER_N_PLAYERS[self.nPlayers]
        self.hands = [ [self._drawTile() for i in range(self.handSize)] for p in range(self.nPlayers) ]

        # players hint, true if an attribute (rank or color) is possible for a tile
        # the attribute in order are [C0, C1, C2, C3, C4, V1, V2, V3, V4, V5]
        self.hints = [[[[True] * Game.N_RANKS, [True] * Game.N_COLORS] for t in range(self.handSize)] for p in range(self.nPlayers)]

        # actions players can take
        self.actions = []
        for i in range(self.handSize):
            ac = Action(ActionType.PLAY)
            ac.targetTile = i
            self.actions.append(ac)
        for i in range(self.handSize):
            ac = Action(ActionType.DISCARD)
            ac.targetTile = i
            self.actions.append(ac)
        for p in range(1, self.nPlayers):
            for r in Game.RANKS:
                ac = Action(ActionType.HINT)
                ac.targetPlayer = p
                ac.hintAttribute = r
                ac.hintIsColor = False
                self.actions.append(ac)
            for c in Game.COLORS:
                ac = Action(ActionType.HINT)
                ac.targetPlayer = p
                ac.hintAttribute = c
                ac.hintIsColor = True
                self.actions.append(ac)
        self.actions.append(Action(ActionType.NONE)) # do nothing

    def isOver(self):
        """
        Returns: True if game is over
        """
        return self.nFuseTokens <= 0 or self.deckSize <= 0 or self.score == 25

    def getValidActions(self):
        """
        Returns a vector of 1 and 0, where 1 indicates that the corresponding 
        action is valid, 0 otherwise.
        """
        valid = [1] * (2 * self.handSize) # play and discard always valid
        # hint
        if self.nHintTokens == 0:
            # cannot hint if out of hint tokens
            valid += [0] * ((self.nPlayers-1) * (Game.N_RANKS + Game.N_COLORS))
            valid.append(0)
            return valid

        # for each player distance p away, check for valid hints
        for p in range(1, self.nPlayers):
            hand = self.hands[ (self.curPlayer+p)%self.nPlayers ]    
            hasRank = [0] * Game.N_RANKS
            hasColor = [0] * Game.N_COLORS
            
            for tile in hand:
                hasRank[tile.rank] = 1
                hasColor[tile.color] = 1

            valid += hasRank
            valid += hasColor
        valid.append(0)
        return valid

    def play(self, action):
        if action.type == ActionType.PLAY:
            self._processPlay(action)
        elif action.type == ActionType.DISCARD:
            self._processDiscard(action)
        elif action.type == ActionType.HINT:
            self._processHint(action)
        else:
            raise ValueError("Expected action type to be 'PLAY', 'DISCARD' or 'HINT'; got '%r'".format(action.type))
        
        # move to next player
        self.curPlayer = (self.curPlayer + 1) % self.nPlayers


    def _processPlay(self, action):
        tile = self.hands[self.curPlayer][action.targetTile]
        # remove the played tile from play
        self.nRemainTiles[tile.id] -= 1
        # draw new tile, init hint for the new tile
        self.hands[self.curPlayer][action.targetTile] = self._drawTile()    
        self.hints[self.curPlayer][action.targetTile] = [[True]*Game.N_RANKS, [True]*Game.N_COLORS]

        if tile.rank == self.fireworks[tile.color]:
            # correct tile placement: add to firework pile
            self.fireworks[tile.color] += 1
            self.score += 1
            if self.fireworks[tile.color] == Game.N_RANKS:
                # if completing full pile of a color, add a hint token
                self.nHintTokens = min(Game.MAX_HINTS, self.nHintTokens+1)
        else:
            # incorrect tile placement: remove a fuse
            self.nFuseTokens -= 1

    def _processDiscard(self, action):
        tile = self.hands[self.curPlayer][action.targetTile]
        # dicard the tile
        self.nRemainTiles[tile.id] -= 1
        # draw new tile and init its hint
        self.hands[self.curPlayer][action.targetTile] = self._drawTile()
        self.hints[self.curPlayer][action.targetTile] = [[True]*Game.N_RANKS, [True]*Game.N_COLORS]
        # gain a new hint
        self.nHintTokens = min(Game.MAX_HINTS, self.nHintTokens+1)


    def _processHint(self, action):
        # remove a hint token
        self.nHintTokens -= 1
        # get target player and attribute
        player = (self.curPlayer + action.targetPlayer) % self.nPlayers
        attribute = action.hintAttribute

        if (action.hintIsColor):
            for i in range(self.handSize):
                match = self.hands[player][i].color == attribute
                tileHint = self.hints[player][i][1]
                for j in range(Game.N_COLORS):
                    if j == attribute:
                        tileHint[j] = match
                    else:
                        tileHint[j] = tileHint[j] and not match
        else:
            for i in range(self.handSize):
                match = self.hands[player][i].rank == attribute
                tileHint = self.hints[player][i][0]
                for j in range(Game.N_RANKS):
                    if j == attribute:
                        tileHint[j] = match
                    else:
                        tileHint[j] = tileHint[j] and not match

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
        #if not self.determinized:
        # choose a random tile in deck
        #r = randint(0, self.deckSize)
        # move the drawn tile to the drawn part
        #self.deck[r], self.deck[self.deckSize] = self.deck[self.deckSize], self.deck[r]
        
        # return the drawn tile
        return self.deck[self.deckSize]  