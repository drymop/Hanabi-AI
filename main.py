from game import Game
from trainers.dqntrainer import Trainer
from utils.attributedict import AttributeDict

if __name__ == '__main__':
    n_players = 3
    game_configs = AttributeDict(
        n_players=n_players,
        n_ranks=Game.N_RANKS,
        n_suits=Game.N_SUITS,
        hand_size=Game.HAND_SIZE_PER_N_PLAYERS[n_players],
    )
    model_configs = AttributeDict(
        n_rnn_hiddens=64,
        n_rnn_layers=3,
        n_dense_before_rnn=3,
        n_dense_after_rnn=3,
        n_outputs=Game.ACTIONS_PER_N_PLAYERS[n_players],
        learn_rate=2e-5,
    )
    train_configs = AttributeDict(
        save_folder='save',
        buffer_size=32768,  # 2^15
        n_games_per_iter=1024,
        n_validation_games_per_iter=128,
        batch_size=128,
        time_steps=32,
        n_epochs_per_iter=64,
        explore_rate=(1, 0.15, 0.01),
        discount_rate=0.97,
        firework_eval=(1,0.5),
        fuse_eval=(0.5,0.1),
    )
    trainer = Trainer(game_configs, model_configs, train_configs)
    trainer.start_training()