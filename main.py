from game import Game
from trainers.dqntrainer import Trainer
from utils.attributedict import AttributeDict

if __name__ == '__main__':
    save_folder = input('Save folder name: ')
    print('Saving into folder: "{}"'.format(save_folder))

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
        learn_rate=1e-4,
        dropout_rate=0.3,
    )
    train_configs = AttributeDict(
        save_folder=save_folder,
        buffer_size=131072,  # 2^17
        n_games_per_iter=1024,
        n_validation_games_per_iter=128,
        update_target_model_every_n_iter=5,
        batch_size=128,
        time_steps=32,
        n_epochs_per_iter=64,
        explore_rate=(1, 0.15, 0.001),
        discount_rate=0.97,
        firework_eval=(1,0.5),
        fuse_eval=(0.3,0.1),
    )
    trainer = Trainer(game_configs, model_configs, train_configs)
    trainer.start_training()
