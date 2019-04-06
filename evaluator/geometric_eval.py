import numpy as np
from game import Game

class Evaluator:
    def __init__(self, train_configs):
        firework_val, firework_inc = train_configs.firework_eval
        self.firework_eval = [0, firework_val]
        for i in range(Game.N_RANKS - 1):
            firework_val += firework_inc
            self.firework_eval.append(self.firework_eval[-1] + firework_val)

        fuse_val, fuse_inc = train_configs.fuse_eval
        self.fuse_eval = [0, fuse_val]
        for i in range(Game.MAX_FUSES - 1):
            fuse_val += fuse_inc
            self.fuse_eval.append(self.fuse_eval[-1] + fuse_val)

        self.firework_eval = np.array(self.firework_eval)
        self.fuse_eval = np.array(self.fuse_eval)

    def eval(self, game_state):
        return sum(self.firework_eval[x] for x in game_state.fireworks) \
               - self.fuse_eval[Game.MAX_FUSES - game_state.n_fuse_tokens]