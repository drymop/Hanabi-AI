import json
import numpy as np

from game import Game
from model.dqnmodel import Model
from trainer.dqntrainer import Trainer
from utils.attributedict import AttributeDict


class DQNPlayer:
    def __init__(self, load_folder: str, iteration: int):
        with open(load_folder + '/configs.json', 'r') as f:
            configs = json.load(f, object_pairs_hook=AttributeDict)
        game_configs = configs.game_configs
        model_configs = configs.model_configs

        self.model = Model(game_configs, model_configs)
        self.model.load_checkpoint(load_folder, filename=Trainer.checkpoint_file_name(iteration))

    def get_action(self, game: Game):
        state = self.model.extract_features(game)[game.cur_player]
        [q_values] = self.model.predict([state])
        max_q = max(q for i, q in enumerate(q_values) if game.is_valid_action[i])
        choices = [i for i, q in enumerate(q_values) if q == max_q and game.is_valid_action[i]]
        return np.random.choice(choices)
