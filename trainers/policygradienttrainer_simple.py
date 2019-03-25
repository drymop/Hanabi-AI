import itertools
import json
import math
import numpy as np
import os
import random
import time
from typing import List, Tuple

from game import Game
from models.policygradientmodel_simple import Model
from utils.consoledisplay import display_action, display_state


class Trainer:

    def __init__(self, game_configs, model_configs, train_configs):
        # -------------------------
        # configs
        self.game_configs = game_configs
        self.model_configs = model_configs
        self.train_configs = train_configs

        n_players = game_configs.n_players
        n_actions = Game.ACTIONS_PER_N_PLAYERS[n_players]

        # -------------------------
        # models to train and experience _buffer

        self.train_model = Model(game_configs, model_configs)
        # model used during training iteration while train_model is being updated
        self.target_model = Model(game_configs, model_configs)

        # self.experience_buffer = ExperienceBuffer(self.train_configs.buffer_size,
        #                                           self.train_configs.buffer_prob_scale)

        # -------------------------
        # Precomputed neural network's inputs

        # various valid masks for a game state
        self.valid_mask_none = np.zeros(n_actions)  # none of the actions are valid
        self.valid_mask_do_nothing = np.zeros(n_actions)  # only the last action is valid (do nothing)
        self.valid_mask_do_nothing[-1] = 1

        # -------------------------
        # Evaluation of game state
        firework_val, firework_inc = self.train_configs.firework_eval
        self.firework_eval = [0, firework_val]
        for i in range(Game.N_RANKS - 1):
            firework_val += firework_inc
            self.firework_eval.append(self.firework_eval[-1] + firework_val)
        fuse_val, fuse_inc = self.train_configs.fuse_eval
        self.fuse_eval = [0, fuse_val]
        for i in range(Game.MAX_FUSES - 1):
            fuse_val += fuse_inc
            self.fuse_eval.append(self.fuse_eval[-1] + fuse_val)

        self.firework_eval = np.array(self.firework_eval)
        self.fuse_eval = np.array(self.fuse_eval)

    @staticmethod
    def one_hot_hint(hint):
        return np.fromiter((1 if b1 and b2 else 0 for b1 in hint[1] for b2 in hint[0]),
                           dtype=np.int8, count=Game.N_TYPES)

    def extract_game_state(self, game: Game, last_action: int = -1) -> Model.StateFeatures:
        """
        Create state from current player's point of view
        """
        n_players = game.n_players
        p = game.cur_player

        remain_tiles = np.fromiter(game.n_tiles_per_type, dtype=np.int8, count=len(game.n_tiles_per_type))
        # all_hints[player]
        all_hands = [np.fromiter((tile.id for tile in hand), dtype=np.int8, count=game.hand_size)
                     for hand in game.hands]
        # all_hints[player][tile]
        all_hints = [[Trainer.one_hot_hint(tile_hint) for tile_hint in player_hint] for player_hint in game.hints]
        fireworks = np.fromiter(game.fireworks, dtype=np.int8, count=len(game.fireworks))

        if game.is_over:
            valid_mask_cur_player = self.valid_mask_none
        else:
            valid_mask_cur_player = np.fromiter((1 if x else 0 for x in game.is_valid_action),
                                                dtype=np.int8, count=game.n_actions)

        game_state = Model.StateFeatures(
            remain_tiles=remain_tiles,
            hands=[all_hands[i % n_players] for i in range(p + 1, p + n_players)],
            hints=[all_hints[i % n_players] for i in range(p, p + n_players)],
            n_hint_tokens=game.n_hint_tokens,
            n_fuse_tokens=game.n_fuse_tokens,
            fireworks=fireworks,
            last_action=last_action,
            valid_mask=valid_mask_cur_player
        )
        return game_state

    def play_batch(self, n_games: int, select_max: bool = False) -> Tuple[List[Game], List[List[Model.StateFeatures]]]:
        """
        Play a batch of games using the train_model neural network to select move
        :param n_games: number of games
        :param select_max: always select the action with largest probability
        :return: list of games played and list of time series
        """
        n_players = self.game_configs.n_players

        # 2d array of state, recording the time series of each player for each game
        # (aka the time series of player j of game i is stored at index (i*n_players + j)
        time_series = [[] for _ in range(n_games)]  # type: List[List[Model.StateFeatures]]
        games = [Game(n_players) for _ in range(n_games)]
        for game in games:  # vary the number of fuse to start with
            game.n_fuse_tokens = random.randrange(1, Game.MAX_FUSES + 1)
        last_actions = [-1] * n_games

        while not all(g.is_over for g in games):
            # extract game state per player per game into each time series
            for i in range(n_games):
                if games[i].is_over:
                    continue
                cur_game_state = self.extract_game_state(games[i], last_actions[i])
                time_series[i].append(cur_game_state)

            # use NN to figure out action prob for each current game state
            cur_game_states = [ts[-1] for ts in time_series]  # last state of each time series
            action_probs = self.train_model.predict(cur_game_states)

            # choose action for each game based on Q values obtained (and exploration rate)
            for i, game in enumerate(games):
                if game.is_over:
                    continue
                action_prob = action_probs[i]
                if select_max:
                    action_id = np.argmax(action_prob)
                else:
                    action_id = np.random.choice(game.n_actions, p=action_prob)
                action = game.actions[action_id]
                game.play(action)
                last_actions[i] = action_id

        # Add the terminal state
        for i, game in enumerate(games):
            time_series[i].append(self.extract_game_state(game, last_actions[i]))
        return games, time_series

    def play_random(self) -> Tuple[Game, List[Model.StateFeatures]]:
        """Play a game randomly and return the episode.

        Faster than calling play with explore_rate=1, as no neural network is involved.
        """
        n_players = self.game_configs.n_players
        game = Game(n_players)
        time_series = []  # 2d array of state, recording the time series of each game
        last_action = -1

        while not game.is_over:
            game_state = self.extract_game_state(game, last_action)
            time_series.append(game_state)

            # choose action randomly
            choices = [i for i, b in enumerate(game.is_valid_action) if b]
            action_ind = random.choice(choices)
            action = game.actions[action_ind]
            game.play(action)
            last_action = action_ind

        # Add the terminal state
        time_series.append(self.extract_game_state(game, last_action))
        return game, time_series

    def train(self, iteration, trajectories: List[List[Model.StateFeatures]], update_target_model=False):
        if update_target_model:
            save_file = Trainer.checkpoint_file_name(iteration)
            save_folder = self.train_configs.save_folder
            self.train_model.save_checkpoint(folder=save_folder, filename=save_file)
            print('Updating target model')
            self.target_model.load_checkpoint(folder=save_folder, filename=save_file)

        # get action and reward for each trajectory
        discount_rate = self.train_configs.discount_rate
        actions = []
        rewards = []
        for trajectory in trajectories:
            values = [self.eval_game_state(state) for state in trajectory]
            rewards.append(Trainer.calculate_discounted_rewards(values, discount_rate))
            actions.append([trajectory[i].last_action for i in range(1, len(trajectory))])
            trajectory.pop()  # remove terminal state

        # train for each epoch
        avg_loss = 0
        n_epochs = self.train_configs.n_epochs_per_iter
        n_traj_per_epoch = math.ceil(len(trajectories) / n_epochs)
        for i in range(0, len(trajectories), n_traj_per_epoch):
            j = i + n_traj_per_epoch
            batch_states = list(itertools.chain(*trajectories[i:j]))
            batch_actions = list(itertools.chain(*actions[i:j]))
            batch_rewards = list(itertools.chain(*rewards[i:j]))
            avg_loss += self.train_model.train(batch_states, batch_actions, batch_rewards)
        return avg_loss / n_epochs

    @staticmethod
    def calculate_discounted_rewards(state_values: List[float],
                                     discount_rate: float,
                                     normalize: bool = True) -> List[float]:
        """
        Calculate the discounted reward after each time step in the trajectory
        :param state_values: value of each state in the trajectory
        :param discount_rate:
        :param normalize: if True, normalize the reward to have mean 0 and stddev 1
        :return: discounted reward after each step in trajectory (length is 1 less than length of state_values)
        """
        rewards = np.empty(len(state_values) - 1, np.float32)
        cur_reward = 0.0
        for i in reversed(range(len(rewards))):
            cur_reward = cur_reward * discount_rate + state_values[i + 1] - state_values[i]
            rewards[i] = cur_reward
        # Normalize
        if normalize:
            mean = np.mean(rewards)
            std = np.std(rewards)
            if std == 0:
                std = 1
            rewards = (rewards - mean) / std
        return rewards

    def start_training(self):
        # -------------------------
        # create folder and files for recording train progress
        save_folder = self.train_configs.save_folder
        stats_file_path = os.path.join(save_folder, 'stats.csv')

        if not os.path.exists(save_folder):
            print("Save directory does not exist! Making directory {}".format(save_folder))
            os.mkdir(save_folder)

        # file recording statistics during training
        with open(stats_file_path, 'a+') as stats_file:
            stats_file.write('iter, sample_score, sample_eval, sample_deaths, sample_turns, '
                             'valid_score, valid_eval, valid_deaths, valid_turns, loss, time\n')

        # file saving the configs
        with open(os.path.join(save_folder, 'configs.json'), 'w+') as configs_file:
            json.dump(dict(game_configs=self.game_configs,
                           model_configs=self.model_configs,
                           train_configs=self.train_configs),
                      configs_file, indent=4)

        # -------------------------
        # start training

        n_sample_games = self.train_configs.n_games_per_iter
        n_validation_games = self.train_configs.n_validation_games_per_iter
        # do iterations 0 -> infinity
        for it in itertools.count():
            start_iter_time = time.time()
            print('===================================== ITER {} ========================================='.format(it))

            # create sample games by playing with exploration on
            games, batch_trajectories = self.play_batch(n_sample_games)
            # print statistics
            sample_score = sum(game.score for game in games) / n_sample_games
            sample_eval = sum(self.eval_game_state(ts[-1]) for ts in batch_trajectories) / len(batch_trajectories)
            sample_deaths = sum(Game.MAX_FUSES - game.n_fuse_tokens for game in games) / n_sample_games
            sample_turns = sum(game.n_turns for game in games) / n_sample_games
            print('\n{} sample games played'.format(n_sample_games))
            print('sample score: {}\nsample eval: {}\nsample deaths: {}\nsample turns: {}'
                  .format(sample_score, sample_eval, sample_deaths, sample_turns))

            # create validation games by playing with exploration off, these games are not used for training
            games, valid_trajectories = self.play_batch(n_validation_games, select_max=True)
            # print statistics
            valid_score = sum(game.score for game in games) / n_validation_games
            valid_eval = sum(self.eval_game_state(ts[-1]) for ts in valid_trajectories) / len(valid_trajectories)
            valid_deaths = sum(Game.MAX_FUSES - game.n_fuse_tokens for game in games) / n_validation_games
            valid_turns = sum(game.n_turns for game in games) / n_validation_games
            print('\n{} validation games played'.format(n_validation_games))
            print('valid score: {}\nvalid eval: {}\nvalid deaths: {}\nvalid turns: {}'
                  .format(valid_score, valid_eval, valid_deaths, valid_turns))

            # train
            update_target_model = it % self.train_configs.update_target_model_every_n_iter == 0
            loss = self.train(it, batch_trajectories, update_target_model=update_target_model)
            print('\nloss: {}'.format(loss))

            iter_total_time = time.time() - start_iter_time
            print('time: {}'.format(iter_total_time))
            # log stuff to stat file
            with open(stats_file_path, 'a+') as stats_file:
                stats = [it,
                         sample_score, sample_eval, sample_deaths, sample_turns,
                         valid_score, valid_eval, valid_deaths, valid_turns,
                         loss, iter_total_time]
                stats_file.write(','.join(map(str, stats)))
                stats_file.write('\n')

    def eval_game_state(self, game_state: Model.StateFeatures):
        return sum(self.firework_eval[x] for x in game_state.fireworks)
        # return sum(self.firework_eval[x] for x in game_state.fireworks) \
        #        - self.fuse_eval[Game.MAX_FUSES - game_state.n_fuse_tokens]
        # return sum(game_state.fireworks) / (Game.MAX_FUSES + 1 - game_state.n_fuse_tokens)

    def test(self):
        n_players = self.game_configs.n_players
        game = Game(n_players)
        last_action = -1

        while not game.is_over:
            game_states = self.extract_game_state(game, last_action)
            [state_q] = self.train_model.predict([game_states[game.cur_player]])

            # display game
            display_state(game, first_person=False)

            # choose best action among the heuristically allowed actions
            for action_id in forbidden_choices:
                state_q[action_id] = -9

            best_q = max(state_q[j] for j in range(game.n_actions) if game.is_valid_action[j])
            choices = [j for j in range(game.n_actions) if state_q[j] == best_q]
            action_id = random.choice(choices)
            action = game.actions[action_id]

            print("Q vector:")
            fm1 = ' '.join(['%5d'] * len(state_q))
            fm2 = ' '.join(['%5.2f'] * len(state_q))
            print(fm2 % tuple(state_q))
            print(fm1 % tuple(range(len(state_q))))

            print("Chosen action is {}:".format(action_id))
            display_action(game, action)

            game.play(action)
            last_action = action_id
            input()

    @staticmethod
    def checkpoint_file_name(iteration):
        return str(iteration) + '.ckpt'
