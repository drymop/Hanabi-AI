from StandardGame import Game
from pprint import pprint

g = Game(3)
PRINT_KEY = ['deckSize', 'hands', 'curPlayer', 'hints', 'nHintTokens', 'fireworkPile', 'nFuseTokens', 'score']

def pp():
  ndict = {k: g.__dict__[k] for k in PRINT_KEY}
  pprint(ndict)

def pl(a):
  g.play(a)
  pp()


pprint(Game.__dict__)
pprint(g.__dict__)
pp()

Action = type('',(),{})

ha = Action()
ha.type = "HINT"
ha.targetPlayer = 1
ha.hintAttribute = 0
ha.hintIsColor = False

pa = Action()
pa.type = "PLAY"
pa.targetTile = 0

da = Action()
da.type = "DISCARD"
da.targetTile = 0

