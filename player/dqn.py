import json

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

    def get_action(self, game):
        pass