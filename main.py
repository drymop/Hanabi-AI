from StandardGame import Game
from pprint import pprint
from ConsoleDisplay import display
from RandomPlayer import getAction
import random

g = Game(3)

PRINT_KEY = ['deckSize', 'hands', 'curPlayer', 'hints', 'nHintTokens', 'fireworkPile', 'nFuseTokens', 'score']
RANKS  = ('1', '2', '3', '4', '5')
COLORS = ('B', 'G', 'R', 'W', 'Y')

# def printAction(action, game):
#   print(action.type, end = " ")
#   if action.type == 'PLAY' or action.type == 'DISCARD':
#     print((action.targetTile+1))
#   elif action.type == 'HINT':
#     print((action.targetPlayer+game.curPlayer+1)%game.N_PLAYERS, end = " ")
#     if action.hintIsColor:
#       print(COLORS[action.hintAttribute])
#     else:
#       print(RANKS[action.hintAttribute])
#   else:
#     print()
def printAction(action, game):
  pprint(action.__dict__)



def pp():
  ndict = {k: g.__dict__[k] for k in PRINT_KEY}
  pprint(ndict)

def dd():
  display(g)

def pl(a):
  g.play(a)
  display(g)



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

while True:
  g = Game(random.randint(2,5))
  while not g.isOver():
    dd()
    a = getAction(g)
    print()
    printAction(a, g)
    if (a.type == 'UNDO'):
      g.undo()
    else:
      g.play(a)
  dd()
  input()