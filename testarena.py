import numpy as np

from arena import play_batch
from game import Game
from player.dqn import DQNPlayer

if __name__ == '__main__':
    Game.MAX_FUSES = 25
    n_players = 3
    n_games = 10000

    load_dir = 'save36_deep25fuse'
    load_iter = 79180
    player = DQNPlayer(load_dir, load_iter)

    print('Player: dqn, dqn, dqn')
    print('Play %d games' % n_games)
    games, action_freqs = play_batch(n_players=n_players, players=player, n_games=n_games)

    turns = [g.n_turns for g in games]
    print('Turn: %.2f +/- %.2f' % (np.average(turns), np.std(turns)))

    scores = [sum(g.fireworks) for g in games]
    print('Score: %.2f +/- %.2f' % (np.average(scores), np.std(scores)))

    deaths = [Game.MAX_FUSES - g.n_fuse_tokens for g in games]
    print('Deaths: %.2f +/- %.2f' % (np.average(deaths), np.std(deaths)))

    print('Action freq:')
    freq = [sum(x) / n_players / n_games for x in zip(*action_freqs)]
    format = '%5.1f ' * len(freq)
    print(format % tuple(freq))

