COLORS = ('B', 'G', 'R', 'W', 'Y')

class Action(object):
	pass

class HumanPlayer:
	def __init__(self):
		pass

	def getAction(self, game_state):
		action = Action()
		while True:
			tokens = input('\nMake a move: [p|d|h]').split()
			try:
				if tokens[0] == 'p':
					# play
					action.type = 'PLAY'
					action.targetTile = int(tokens[1]) - 1
					if action.targetTile >= game_state.HAND_SIZE or action.targetTile < 0:
						raise ValueError()
				elif tokens[0] == 'd':
					# discard
					action.type = 'DISCARD'
					action.targetTile = int(tokens[1])
					if action.targetTile >= game_state.HAND_SIZE or action.targetTile < 0:
						raise ValueError()
				elif tokens[0] == 'h':
					action.type = 'HINT'
					action.targetPlayer = (int(tokens[1]) - game_state.curPlayer - 1) % game_state.N_PLAYERS
					try:
						action.hintAttribute = int(tokens[2]) - 1
						action.hintIsColor = False
						if action.hintAttribute < 0 or action.hintAttribute >= game_state.N_RANKS:
							raise ValueError()
					except Exception as e:
						action.hintAttribute = COLORS.index(tokens[2])
						action.hintIsColor = True 
				else:
				 raise ValueError()
			except Exception as e:
				pass
			else:
				return action

