from Model import Model
import numpy as np
from pprint import pprint as pp
import collections

N_PLAYERS = 4;
N_TILE_TYPES = 25;
HAND_SIZE = 4;
N_COLORS = 5;
N_RANKS  = 5
MAX_HINT_TOKENS = 8
MAX_FUSE_TOKENS = 3
MAX_N_PER_TYPES = 3

players = [1, 0, 0 , 0]
rem = [1] * N_TILE_TYPES
hands = [[[0] * N_TILE_TYPES for _ in range(HAND_SIZE)] for _ in range(N_PLAYERS-1)]
hints = [[[1] * N_TILE_TYPES for _ in range(HAND_SIZE)] for _ in range(N_PLAYERS)]
hint_tok = 8
fuse_tok = 3
fireworks = [1, 3, 2, 0, 3]
initial_state = np.zeros((2, 51))

# test prediction
Game = collections.namedtuple('Game', 'cur_player remain_types hands hints hint_tokens fuse_tokens fireworks initial_state')
g = Game([players], [rem], [hands], [hints], [hint_tok], [fuse_tok], [fireworks], initial_state)

m = Model(51, 32)

v, s = m.predict(g)
pp(v)
pp(s)

# test training
Games = collections.namedtuple('Games', 'cur_player remain_types hands hints hint_tokens fuse_tokens fireworks initial_state labels output_mask')
labels = [[[4] * 32] * 5]
output_mask = np.empty((1, 5, 32))
output_mask.fill(1)
g2 = Games([[players]*5], [[rem]*5], [[hands]*5], [[hints]*5], [[hint_tok]*5], [[fuse_tok]*5], [[fireworks]*5], [initial_state], labels, output_mask)
loss = m.train(g2)
pp(loss)

print('================================================================================')
v, s = m.predict(g)
pp(v)
pp(s)

print('================================================================================')
print('Save and load')
m.save_checkpoint()

m = Model(51, 32)
m.load_checkpoint()
v, s = m.predict(g)
pp(v)
pp(s)