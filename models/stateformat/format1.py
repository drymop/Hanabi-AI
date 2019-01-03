class StateFormatter:
    """Format a game state into input for a neural network"""

    def __init__(self, nPlayers, batchSize, timeStep, nActions):
        # precompute onehot vectors
        self.onehotPlayers = np.identity(nPlayers)
        self.onehotTiles = np.identity(Game.N_TYPES)

        # various loss masks for a state
        self.maskOnes = np.ones(nActions)
        self.maskZeros = np.zeros(nActions)
        
        # valid action masks
        self.maskNonCurPlayer = np.zeros(nActions)
        self.maskNonCurPlayer[-1] = 1