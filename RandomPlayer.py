import random

class Action(object):
  pass

TYPE = ['PLAY'] * 3 + ['DISCARD'] * 2  + ['HINT'] * 5 + ['UNDO']

def getAction(game):
  action = Action()
  action.type = random.choice(TYPE)
  if action.type == 'PLAY' or action.type == 'DISCARD':
    action.targetTile = random.randrange(game.HAND_SIZE)
  elif action.type == 'HINT':
    action.targetPlayer = random.randrange(1, game.N_PLAYERS)
    action.hintAttribute = random.randrange(game.N_COLORS)
    action.hintIsColor = random.choice([True, False])
  
  return action