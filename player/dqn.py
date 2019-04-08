import json
import numpy as np
from typing import List

from game import Game
from model.dqnmodel import Model
from player.base import Player
from trainer.dqntrainer import Trainer
from utils.attributedict import AttributeDict


class DQNPlayer(Player):
    def __init__(self, model: Model = None, load_folder: str = '', iteration: int = -1):
        if model:
            self.model = model
        else:
            # load from file
            with open(load_folder + '/configs.json', 'r') as f:
                configs = json.load(f, object_pairs_hook=AttributeDict)
            game_configs = configs.game_configs
            model_configs = configs.model_configs

            self.model = Model(game_configs, model_configs)
            self.model.load_checkpoint(load_folder, filename=Trainer.checkpoint_file_name(iteration))

    def get_action(self, game: Game) -> int:
        state = self.model.extract_features(game)[game.cur_player]
        [q_values] = self.model.predict([state])
        max_q = max(q for i, q in enumerate(q_values) if game.is_valid_action[i])
        choices = [i for i, q in enumerate(q_values) if q == max_q and game.is_valid_action[i]]
        return np.random.choice(choices)

    def get_batch_actions(self, games: List[Game]) -> List[int]:
        batch_state = [self.model.extract_features(game)[game.cur_player] for game in games]
        batch_q = self.model.predict(batch_state)
        batch_action = [self.select_action_from_q(game, q_values) for game, q_values in zip(games, batch_q)]
        return batch_action

    @staticmethod
    def select_action_from_q(game: Game, q_values: List[float]) -> int:
        max_q = max(q for i, q in enumerate(q_values) if game.is_valid_action[i])
        choices = [i for i, q in enumerate(q_values) if q == max_q and game.is_valid_action[i]]
        if not choices:
            return -1
        return np.random.choice(choices)
