from pprint import pprint
from StandardGame import ActionType

RANKS  = ('1', '2', '3', '4', '5')
COLORS = ('B', 'G', 'R', 'W', 'Y')

def displayState(game, firstPerson=True):
  print("Start")
  n = game.nPlayers
  s = game.handSize

  print('-' * 120)

  gapLen  = 7
  handLen = s * 7 - 2

  sep    = '-' * handLen
  gap    = ' ' * gapLen
  sepRow = gap.join([sep] * n)

  tileFormat = ' %s%s  '
  tileRow = '  '.join([tileFormat] * s)
  tileRow = gap.join([tileRow] * n)

  # print hints and fuses
  print('Player   : %d \nHints    : %d \nFuses    : %d\n' % (game.curPlayer+1, game.nHintTokens, game.nFuseTokens))

  # print firework pile
  pileStr = '  '.join(['%d%s'] * game.N_COLORS)
  pile    = sum( ((game.fireworks[col] , COLORS[col]) for col in range(0, game.N_COLORS)), () )
  print('Firework : ', end = "")
  print(pileStr % pile)
  print()


  # print the tiles players' hands
  hands = sum( ([RANKS[t.rank], COLORS[t.color]] for p in range(0, n) for t in game.hands[p]), [] )
  if firstPerson:
    twoS = 2 * s
    hands[:] = [hands[i] if (i // twoS != game.curPlayer) else '?' for i in range(0, len(hands)) ]
  hands = tuple(hands)

  print(sepRow)
  print(tileRow % hands)
  print(sepRow)

  # print the hints
  hintStr = '  '.join(['%s' * game.N_COLORS] * s)
  hintStr = gap.join([hintStr] * n)
  hints = [RANKS[i] if game.hints[p][t][0][i] else '_' for p in range(0, n) for t in range(0, s) for i in range(0, game.N_RANKS)]  
  print(hintStr % tuple(hints))
  hints = [COLORS[i] if game.hints[p][t][1][i] else '_' for p in range(0, n) for t in range(0, s) for i in range(0, game.N_COLORS)]  
  print(hintStr % tuple(hints))
  print("End")


def displayAction(game, action):
  print('Action: ', end="")
  if action.type == ActionType.PLAY:
    tile = game.hands[game.curPlayer][action.targetTile]
    print('Play tile {} ({}{})'.format(action.targetTile+1, RANKS[tile.rank], COLORS[tile.color]))
  elif action.type == ActionType.DISCARD:
    tile = game.hands[game.curPlayer][action.targetTile]
    print('Discard tile {} ({}{})'.format(action.targetTile+1, RANKS[tile.rank], COLORS[tile.color]))
  elif action.type == ActionType.HINT:
    attr = COLORS[action.hintAttribute] if action.hintIsColor else RANKS[action.hintAttribute]
    target = (game.curPlayer + action.targetPlayer) % game.nPlayers + 1
    print('Hint player {} of all {} tiles'.format(target, attr))
  else:
    print('Unknown action')