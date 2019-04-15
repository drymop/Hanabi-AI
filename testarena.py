import numpy as np

from arena import play_batch
from game import Game
from player.dqn import DQNPlayer

if __name__ == '__main__':
    Game.MAX_FUSES = 3
    n_players = 3
    n_games = 10000

    load_dir = 'save39_high3fusepen'
    load_iter = 33710
    player = DQNPlayer(load_folder=load_dir, file_name=str(load_iter)+'.ckpt')

    print('Player: dqn')
    print('Max fuses: %d' % Game.MAX_FUSES)
    games, action_freqs = play_batch(n_players=n_players, players=player, n_games=n_games)

    turns = [g.n_turns for g in games]
    print('Turn: %.2f +/- %.2f' % (np.average(turns), np.std(turns)))

    scores = [sum(g.fireworks) for g in games]
    print('Score: %.2f +/- %.2f' % (np.average(scores), np.std(scores)))

    deaths = [Game.MAX_FUSES - g.n_fuse_tokens for g in games]
    print('Deaths: %.2f +/- %.2f' % (np.average(deaths), np.std(deaths)))

    print('Action freq:')
    freq = [sum(x) / n_players / n_games for x in zip(*action_freqs)]
    ind_format = '%4d' * len(freq)
    print(ind_format % tuple(i for i in range(len(freq))))
    format = '%4.1f' * len(freq)
    print(format % tuple(freq))

