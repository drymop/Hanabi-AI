import itertools
import json
import numpy as np
import os
import random
import time

from game import Game
from models.dqnmodel import Model
from utils.expbuffer import Experience, ExperienceBuffer


class Trainer:

    def __init__(self, game_configs, model_configs, train_configs):
        # configs
        self.game_configs = game_configs
        self.model_configs = model_configs
        self.train_configs = train_configs

        n_players = game_configs.n_players
        n_actions = Game.ACTIONS_PER_N_PLAYERS[n_players]

        n_rnn_layers = model_configs.n_rnn_layers
        n_rnn_hiddens = model_configs.n_rnn_hiddens

        batch_size = train_configs.batch_size
        time_steps = train_configs.time_steps

        # -------------------------
        # models to train and experience buffer

        self.train_model = Model(game_configs, model_configs)
        # model used during training iteration while train_model is being updated
        self.target_model = Model(game_configs, model_configs)

        self.experience_buffer = ExperienceBuffer(self.train_configs.buffer_size)

        # -------------------------
        # Precomputed neural network's inputs

        # various valid masks for a game state
        self.valid_mask_none = np.zeros(n_actions)  # none of the actions are valid
        self.valid_mask_do_nothing = np.zeros(n_actions)  # only the last action is valid (do nothing)
        self.valid_mask_do_nothing[-1] = 1

        # loss masks
        self.loss_mask_batch_all = np.ones(shape=(batch_size, time_steps, n_actions))

        # rnn zero state for a batch
        self.rnn_zero_state_batch = np.zeros((n_rnn_layers, 2, batch_size, n_rnn_hiddens))

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

    def extract_game_state(self, game, last_action):
        """
        Create n states (n = n_players), each is a game state from each player's point of view
        """
        n_players = game.n_players
        remain_tiles = np.fromiter(game.n_tiles_per_type, dtype=np.int8, count=len(game.n_tiles_per_type))
        # all_hints[player]
        all_hands = [np.fromiter((tile.id for tile in hand), dtype=np.int8, count=game.hand_size)
                     for hand in game.hands]
        # all_hints[player][tile]
        all_hints = [[Trainer.one_hot_hint(tile_hint) for tile_hint in player_hint] for player_hint in game.hints]
        fireworks = np.fromiter(game.fireworks, dtype=np.int8, count=len(game.fireworks))

        if game.is_over:
            valid_mask_cur_player = self.valid_mask_none
            valid_mask_other_player = self.valid_mask_none
        else:
            valid_mask_cur_player = np.fromiter((1 if x else 0 for x in game.is_valid_action),
                                                dtype=np.int8, count=game.n_actions)
            valid_mask_other_player = self.valid_mask_do_nothing

        game_states = [None] * n_players
        for p in range(n_players):
            game_states[p] = Model.StateFeatures(
                cur_player=(game.cur_player - p) % n_players,
                remain_tiles=remain_tiles,
                hands=[all_hands[i % n_players] for i in range(p + 1, p + n_players)],
                hints=[all_hints[i % n_players] for i in range(p, p + n_players)],
                n_hint_tokens=game.n_hint_tokens,
                n_fuse_tokens=game.n_fuse_tokens,
                fireworks=fireworks,
                last_action=last_action,
                valid_mask=valid_mask_cur_player if p == game.cur_player else valid_mask_other_player,
            )
        return game_states

    def get_train_batch(self, batch_size, time_steps):
        """Return an array of game states with shape (batch_size, time_steps), chosen randomly from the experience
        buffer
        """
        batch = self.experience_buffer.sample(batch_size)
        for i in range(batch_size):
            # trim the series to time_steps steps
            series = batch[i].experience
            r = random.randrange(len(series) - time_steps)
            batch[i] = series[r:r + time_steps + 1]
        return batch

    def play_batch(self, n_games, explore_rate=0.1):
        """
        Play a batch of games using the train_model neural network to select move
        :param n_games: number of games
        :param explore_rate: probability that a move is chosen randomly (instead of choosing the best move)
        :return: list of games played and list of time series
        """
        n_players = self.game_configs.n_players
        time_steps = self.train_configs.time_steps
        batch_size = n_games * n_players

        # 2d array of state, recording the time series of each player for each game
        # (aka the time series of player j of game i is stored at index (i*n_players + j)
        time_series = [[] for _ in range(batch_size)]
        games = [Game(n_players) for _ in range(n_games)]
        rnn_state = np.zeros((self.model_configs.n_rnn_layers, 2, batch_size, self.model_configs.n_rnn_hiddens))
        last_actions = [-1] * n_games

        while not all(g.is_over for g in games):
            # extract game state per player per game into each time series
            for i in range(n_games):
                if games[i].is_over:
                    continue
                cur_game_states = self.extract_game_state(games[i], last_actions[i])
                for j in range(n_players):
                    time_series[i * n_players + j].append(cur_game_states[j])

            # use NN to figure out next move for each game
            cur_game_states = [[ts[-1]] for ts in time_series]  # shape=(batch_size, time_step=1)
            nn_inputs = Trainer.format_batch(cur_game_states)
            batch_q, rnn_state = self.train_model.predict(nn_inputs, rnn_state)

            # choose action for each game based on Q values
            for i, game in enumerate(games):
                if game.is_over:
                    continue
                [action_qs] = batch_q[i * n_players + game.cur_player]
                if random.random() > explore_rate:
                    # choose best action
                    best_q = max(action_qs[j] for j in range(game.n_actions) if game.is_valid_action[j])
                    choices = [j for j in range(game.n_actions) if action_qs[j] == best_q]
                else:
                    # choose a random action
                    choices = [j for j in range(game.n_actions) if game.is_valid_action[j]]
                action_id = random.choice(choices)
                action = game.actions[action_id]
                game.play(action)

                last_actions[i] = action_id

        # Add the terminal state, as well as pad to minimum required length for each time series
        for i, game in enumerate(games):
            n_turns = len(time_series[i * n_players])
            n_pad = max(0, time_steps - n_turns) + 1
            terminal_states = self.extract_game_state(game, last_actions[i])
            for j in range(n_players):
                for _ in range(n_pad):
                    time_series[i * n_players + j].append(terminal_states[j])
        return games, time_series

    def play_random(self):
        """Play a game randomly and return the episode.

        Faster than calling play with explore_rate=1, as no neural network is involved.
        """
        n_players = self.game_configs.n_players
        game = Game(n_players)
        time_series = [[] for _ in range(n_players)]  # 2d array of state, recording the time series of each player
        time_steps = self.train_configs.time_steps
        last_action = -1

        while not game.is_over:
            game_states = self.extract_game_state(game, last_action)
            for p in range(game.n_players):
                time_series[p].append(game_states[p])

            # choose action randomly
            choices = [i for i, b in enumerate(game.is_valid_action) if b]
            action_ind = random.choice(choices)
            action = game.actions[action_ind]
            game.play(action)
            last_action = action_ind

        # Add the terminal state, as well as pad to minimum required length
        n_pad = max(0, time_steps - game.n_turns) + 1
        terminal_states = self.extract_game_state(game, last_action)
        for p in range(game.n_players):
            for _ in range(n_pad):
                time_series[p].append(terminal_states[p])

        return game, time_series

    def train(self, iteration, n_epochs, update_target_model=False):
        save_file = Trainer.checkpoint_file_name(iteration)
        save_folder = self.train_configs.save_folder
        self.train_model.save_checkpoint(folder=save_folder, filename=save_file)
        if update_target_model:
            print('Updating target model')
            self.target_model.load_checkpoint(folder=save_folder, filename=save_file)

        batch_size = self.train_configs.batch_size
        time_steps = self.train_configs.time_steps
        discount_rate = self.train_configs.discount_rate

        # train for some epochs
        avg_loss = 0
        for epoch in range(n_epochs):
            # get experiences
            batch = self.get_train_batch(batch_size, time_steps)  # each feature has shape [batch_size, time_steps+1, ?]

            # reward from time i to i+1:
            batch_rewards = []
            for time_series in batch:
                series_rewards = [None] * time_steps
                cur_eval = self.eval_game_state(time_series[0])
                for i in range(time_steps):
                    next_eval = self.eval_game_state(time_series[i + 1])
                    series_rewards[i] = next_eval - cur_eval
                    cur_eval = next_eval
                batch_rewards.append(series_rewards)

            # Q values for each game state from the batch, shape=(batch_size, time_steps+1, n_actions)
            q_values, _ = self.target_model.predict(Trainer.format_batch(batch), self.rnn_zero_state_batch)

            # target Q values = reward + max(Q values of next state)
            for i in range(batch_size):
                for j in range(time_steps):
                    action, reward = batch[i][j + 1].last_action, batch_rewards[i][j]
                    q_values[i][j][action] = reward + discount_rate * max(q_values[i][j + 1])

            # remove the last time step
            for series in batch:
                series.pop()
            q_values = np.delete(q_values, -1, axis=1)
            loss = self.train_model.train(Trainer.format_batch(batch),
                                          self.rnn_zero_state_batch, q_values, self.loss_mask_batch_all)
            avg_loss += loss
        return avg_loss / n_epochs

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
            stats_file.write('iter, explore_rate, buffer_eval, sample_score, sample_eval, sample_deaths, '
                             'sample_turns, valid_score, valid_eval, valid_deaths, valid_turns, loss, time\n')

        # file saving the configs
        with open(os.path.join(save_folder, 'configs.json'), 'w+') as configs_file:
            json.dump(dict(game_configs=self.game_configs,
                           model_configs=self.model_configs,
                           train_configs=self.train_configs),
                      configs_file, indent=4)

        # -------------------------
        # fill up the buffer
        print('==================================== FILLING BUFFER ===================================')
        avg_score = 0
        avg_eval = 0
        avg_turns = 0
        n_random_games = self.train_configs.buffer_size // self.game_configs.n_players // 2  # fill half of the buffer
        # n_random_games = 1

        for i in range(n_random_games):
            game, time_series = self.play_random()
            # add to buffer
            _eval = self.eval_game_state(time_series[0][-1])  # evaluation of the final state
            for time_series_a_player in time_series:
                self.experience_buffer.add(Experience(_eval, time_series_a_player), weighted=True)
            # update average
            avg_score += game.score
            avg_eval += _eval
            avg_turns += game.n_turns
            if ((i + 1) % 200) == 0:
                games_played = i + 1
                print('{} games played'.format(games_played))
                print('buffer eval: {}'.format(self.experience_buffer.avgScore))
                print('score: {}'.format(avg_score / games_played))
                print('eval : {}'.format(avg_eval / games_played))
                print('turn : {}'.format(avg_turns / games_played))
        print('score: {}'.format(avg_score / n_random_games))
        print('eval : {}'.format(avg_eval / n_random_games))
        print('turn : {}'.format(avg_turns / n_random_games))

        # -------------------------
        # start training

        explore_rate_start, explore_rate_end, explore_rate_decrease = self.train_configs.explore_rate
        explore_rate = explore_rate_start
        n_sample_games = self.train_configs.n_games_per_iter
        n_validation_games = self.train_configs.n_validation_games_per_iter
        # do iterations 0 -> infinity
        for it in itertools.count():
            start_iter_time = time.time()
            print('===================================== ITER {} ========================================='.format(it))
            print('explore rate: {}'.format(explore_rate))

            # create sample games by playing with exploration on
            games, batch_time_series = self.play_batch(n_sample_games, explore_rate=explore_rate)
            # add sample games to experience buffer
            sample_eval = 0
            for time_series in batch_time_series:
                _eval = self.eval_game_state(time_series[-1])
                sample_eval += _eval
                self.experience_buffer.add(Experience(_eval, time_series), weighted=True)
            # print statistics
            sample_score = sum(game.score for game in games) / n_sample_games
            sample_eval = sample_eval / len(batch_time_series)
            sample_deaths = sum(Game.MAX_FUSES - game.n_fuse_tokens for game in games) / n_sample_games
            sample_turns = sum(game.n_turns for game in games) / n_sample_games
            print('\n{} sample games played'.format(n_sample_games))
            print('buffer eval: {}\nsample score: {}\nsample eval: {}\nsample deaths: {}\nsample turns: {}'
                  .format(self.experience_buffer.avgScore, sample_score, sample_eval, sample_deaths,
                          sample_turns))

            # create validation games by playing with exploration off, these games are not added to experience buffer
            games, batch_time_series = self.play_batch(n_validation_games, explore_rate=0)
            # print statistics
            valid_score = sum(game.score for game in games) / n_validation_games
            valid_eval = sum(self.eval_game_state(ts[-1]) for ts in batch_time_series) / len(batch_time_series)
            valid_deaths = sum(Game.MAX_FUSES - game.n_fuse_tokens for game in games) / n_validation_games
            valid_turns = sum(game.n_turns for game in games) / n_validation_games
            print('\n{} validation games played'.format(n_validation_games))
            print('valid score: {}\nvalid eval: {}\nvalid deaths: {}\nvalid turns: {}'
                  .format(valid_score, valid_eval, valid_deaths, valid_turns))

            # train
            update_target_model = it % self.train_configs.update_target_model_every_n_iter == 0
            loss = self.train(it, self.train_configs.n_epochs_per_iter, update_target_model=update_target_model)
            print('\nloss: {}'.format(loss))

            iter_total_time = time.time() - start_iter_time
            print('time: {}'.format(iter_total_time))
            # log stuff to stat file
            with open(stats_file_path, 'a+') as stats_file:
                stats = [it, explore_rate, self.experience_buffer.avgScore,
                         sample_score, sample_eval, sample_deaths, sample_turns,
                         valid_score, valid_eval, valid_deaths, valid_turns,
                         loss, iter_total_time]
                stats_file.write(','.join(map(str, stats)))
                stats_file.write('\n')

            # update iteration related variables
            explore_rate = max(explore_rate - explore_rate_decrease, explore_rate_end)

    def eval_game_state(self, game_state):
#        return sum(self.firework_eval[x] for x in game_state.fireworks) \
#               - self.fuse_eval[Game.MAX_FUSES - game_state.n_fuse_tokens]
        return sum(game_state.fireworks) / (Game.MAX_FUSES + 1 - game_state.n_fuse_tokens)

    @staticmethod
    def checkpoint_file_name(iteration):
        return str(iteration) + '.ckpt'

    @staticmethod
    def format_batch(time_series_batch):
        """
        Create a batch from a list of time series, each time series is a list of States
        (aka a 2D array of game states, shape=[batch_size x time_steps])
        The resulting batch collects each feature of game state in a 2D array of shape [batch_size x time_steps],
        where element[i][j] is the feature of State at batch i, time j
        """
        return Model.StateFeatures(*zip(*(zip(*s) for s in time_series_batch)))
