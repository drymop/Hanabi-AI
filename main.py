from trainer.dqntrainer import Trainer
from utils.attributedict import AttributeDict

if __name__ == '__main__':
    save_folder = input('Save folder name: ')
    print('Saving into folder: "{}"'.format(save_folder))

    n_players = 3
    game_configs = AttributeDict(
        n_players=n_players,
        max_fuses=3,
    )
    model_configs = AttributeDict(
        load=AttributeDict(folder='save39_high3fusepen', file='33710.ckpt'),
        # n_rnn_hiddens=64,
        # n_rnn_layers=1,
        # n_dense_before_rnn=1,
        # n_dense_after_rnn=1,
        n_hiddens=[512, 256, 128, 64],
        learn_rate=3e-5,
        dropout_rates=[0, 0.3, 0.2, 0.1],
    )
    train_configs = AttributeDict(
        save_folder=save_folder,
        buffer_size=16384,
        buffer_prob_scale=5,
        n_fill_buffer=0,
        n_games_per_iter=512,
        n_validation_games_per_iter=128,
        update_target_model_every_n_iter=10,
        batch_size=128,
        # time_steps=8,
        n_epochs_per_iter=128,
        explore_rate=(0.5, 0.1, 0.001),
        guide_rate=(0, 0, 0),
        discount_rate=0.95,
        firework_eval=(1, 0.5),
        fuse_eval=(2.3, 0.3),
        hint_eval=5,
    )

    trainer = Trainer(game_configs, model_configs, train_configs)
    trainer.start_training()
