import numpy as np
import os
import random
from pprint import pprint as pp

from game import Game
from models.dqnmodel import Model
from utils.attributedict import AttributeDict as attdict
from utils.consoledisplay import display_state, display_action
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

        # -------------------------
        # models to train and experience buffer

        self.train_model = Model(model_configs, game_configs)
        self.targetModel = Model(model_configs, game_configs)
        self.buffer_size = 32768  # 2^15
        self.experience_buffer = ExperienceBuffer(self.buffer_size)

        # -------------------------
        # Precomputed neural network's inputs

        # various valid_mask for a game state
        self.valid_mask_none = np.zeros(n_actions)  # none of the actions are valid
        self.valid_mask_do_nothing = np.zeros(n_actions)  # only the last action is valid (do nothing)
        self.valid_mask_do_nothing[-1] = 1

        # rnn zero state for a batch
        self.rnn_zero_state_batch = np.zeros((n_rnn_layers, 2, batch_size, n_rnn_hiddens))

        # -------------------------
        # Evaluation of game state
        self.firework_eval = np.array([0, 1, 2.2, 3.6, 5.3, 7.3])
        fuse_value = 0.5
        fuse_value_increase = 0.2
        self.fuse_eval = [0, fuse_value]
        for i in range(50):
            fuse_value += fuse_value_increase
            self.fuse_eval.append(self.fuse_eval[-1] + fuse_value)

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
                cur_player=(p - game.cur_player) % n_players,
                remain_tiles=remain_tiles,
                hands=[all_hands[i % n_players] for i in range(p + 1, p + n_players)],
                hints=[all_hints[i % n_players] for i in range(p, p + n_players)],
                hint_tokens=game.n_hint_tokens,
                fuse_tokens=game.n_fuse_tokens,
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

    # def play(self, game, minLength, explore_rate=0.1, show=False):
    #     # 2d array of state, recording the time series of each player
    #     timeSeries = [[] for _ in range(game.n_players)]
    #
    #     rnnStateTuple = self.gameZeroState
    #     score = self.eval_game(game)
    #     while not game.is_over:
    #
    #         game_states = self.extract_game_state(game)
    #         for player in range(game.n_players):
    #             timeSeries[player].append(game_states[player])
    #
    #         x = Trainer.formatBatch([s] for s in game_states)
    #
    #         batchQValues, rnnStateTuple = self.train_model.predict(x, rnnStateTuple)
    #
    #         # get the q-prediction of the current player at the only time step
    #         qValues = batchQValues[game.curPlayer][-1]
    #
    #         validActions = game.getValidActions()
    #
    #         if random.random() > explore_rate:
    #             # choose action with max q value
    #             m = max(qValues[i] for i in range(len(qValues)) if validActions[i])
    #             choices = [i for i in range(len(qValues)) if qValues[i] == m]
    #         else:
    #             # choose action randomly
    #             choices = [i for i, b in enumerate(validActions) if b]
    #         actionInd = random.choice(choices)
    #         action = game.actions[actionInd]
    #
    #         # save the action
    #         game_states[game.curPlayer].action_and_reward[0] = actionInd
    #
    #         if show:
    #             display_state(game, firstPerson=False)
    #             pp(batchQValues)
    #
    #             print('\nAction: {}'.format(actionInd))
    #             display_action(game, action)
    #             print()
    #             input('Press ENTER...')
    #
    #         game.play(action)
    #
    #         # get reward
    #         newScore = self.eval_game(game)
    #         for state in game_states:
    #             state.action_and_reward[1] = newScore - score
    #         score = newScore
    #
    #     if show:
    #         display_state(game, False)
    #         print(game.score, end=" ")
    #         print(game.nFuseTokens)
    #
    #     # Add the terminal state, as well as pad to minimum required length
    #     nTurn = len(timeSeries[0])
    #     pad = max(0, minLength - nTurn) + 1
    #     terminalStates = self.extract_game_state(game)
    #     for _ in range(pad):
    #         for player in range(game.n_players):
    #             timeSeries[player].append(terminalStates[player])
    #
    #     return score, nTurn, timeSeries

    def play_batch(self, n_games, explore_rate=0.1):
        n_players = self.game_configs.n_players
        time_steps = self.train_configs.time_steps
        batch_size = n_games * n_players

        # 2d array of state, recording the time series of each player for each game
        # (aka the time series of player j of game i is stored at index (i*n_players + j)
        time_series = [[] for _ in range(batch_size)]
        games = [Game(n_players) for _ in range(n_games)]
        rnn_state = np.zeros((self.model_configs.n_rnn_layers, 2, batch_size, self.model_configs.n_hiddens))
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
            nn_inputs = Trainer.formatBatch(cur_game_states)
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

                for j in range(i * n_players, (i + 1) * n_players):
                    last_actions[j] = action_id

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

    # def train(self, iteration, epoch, checkpoint_folder='save', discount_rate=0.99):
    #     saveFile = Trainer.checkpointFile(iteration)
    #     self.train_model.save_checkpoint(folder=checkpoint_folder, filename=saveFile)
    #     self.targetModel.load_checkpoint(folder=checkpoint_folder, filename=saveFile)
    #
    #     # train for some epochs
    #     avgLoss = 0
    #     for epochI in range(epoch):
    #         # print('EPOCH ::: ' + str(epochI + 1))
    #
    #         # get experiences
    #         batch = self.get_train_batch(self.batch_size, self.time_steps)
    #
    #         # Q value for such experience
    #         targetQ, _ = self.targetModel.predict(Trainer.formatBatch(batch), self.batchZeroState)
    #
    #         # train against the target
    #         for i in range(self.batch_size):
    #             for j in range(self.time_steps):
    #                 actionTaken, reward = batch[i][j].action_and_reward
    #                 targetQ[i][j][actionTaken] = reward + discount_rate * max(targetQ[i][j + 1])
    #
    #         # remove the last time step
    #         for series in batch:
    #             series.pop()
    #         targetQ = np.delete(targetQ, -1, 1)
    #         t = Trainer.formatBatch(batch)
    #
    #         loss = self.train_model.train(Trainer.formatBatch(batch), self.batchZeroState, targetQ)
    #         print(loss)
    #         avgLoss += loss
    #     return avgLoss / epoch
    #
    # def start(self, startIteration=0, startExploreRate=1, endExploreRate=0.05, exploreDecrease=0.015):
    #     # fill up the buffer
    #     print('==================================== FILLING BUFFER ===================================')
    #     avgScore = 0
    #     avgTurn = 0
    #     # n_random_games = self.buffer_size//self.n_players//2
    #     n_random_games = 1
    #     i = 0
    #     while i < n_random_games:
    #         game = Game(self.n_players)
    #         score, nTurn, timeSeries = self.play_random(game, self.time_steps)
    #
    #         i += 1
    #         for s in timeSeries:
    #             self.experience_buffer.add(Experience(score, s), weighted=True)
    #         avgScore += score
    #         avgTurn += nTurn
    #         if (i % 200) == 0:
    #             print('{} games played'.format(i))
    #             print('buffer score: {}'.format(self.experience_buffer.avgScore))
    #             print('score: {}'.format(avgScore / i))
    #             print('turn : {}'.format(avgTurn / i))
    #     print('score: {}'.format(avgScore / n_random_games))
    #     print('turn : {}'.format(avgTurn / n_random_games))
    #
    #     explore_rate = max(startExploreRate - (startIteration - 1) * exploreDecrease, endExploreRate)
    #
    #     # create folder and files for recording train progress
    #     save_folder = 'save'
    #     stats_file_name = 'save/stats.csv'
    #     if not os.path.exists(save_folder):
    #         print("Save directory does not exist! Making directory {}".format(save_folder))
    #         os.mkdir(save_folder)
    #
    #     with open(stats_file_name, 'a+') as stats_file:
    #         stats_file.write(
    #             'iter, explore_rate, buffer_score, sample_discarded, sample_score, sample_game_score, sample_deaths, sample_turns, validation_score, validation_game_score, validation_deaths, validation_turns, loss\n')
    #
    #     # start training
    #     for it in range(startIteration, 100000 + startIteration):
    #         print('========================================= ITER {} ==================================='.format(it))
    #         print('explore rate: {}'.format(explore_rate))
    #
    #         # create sample games by playing with exploration on
    #         sample_discarded = 0
    #         sample_score = 0
    #         sample_game_score = 0
    #         sample_deaths = 0
    #         sample_turns = 0
    #         sample_games = 500
    #         i = 0
    #         while i < sample_games:
    #             game = Game(self.n_players)
    #             score, nTurn, timeSeries = self.play(game, self.time_steps, explore_rate=explore_rate)
    #             if game.score <= 0:
    #                 sample_discarded += 1
    #                 continue
    #             for s in timeSeries:
    #                 self.experience_buffer.add(Experience(score, s), weighted=True)
    #             sample_score += score
    #             sample_game_score += game.score
    #             sample_deaths += Game.MAX_FUSES - game.n_fuse_tokens
    #             sample_turns += nTurn
    #             if (i % 100) == 0:
    #                 print('{} sample games played, discarded {}'.format(i, sample_discarded))
    #             i += 1
    #
    #         sample_score /= sample_games
    #         sample_game_score /= sample_games
    #         sample_deaths /= sample_games
    #         sample_turns /= sample_games
    #
    #         print('buffer score: {}\nsample score: {}\nsample game score: {}\nsample deaths: {}\nsample turns: {}'
    #               .format(self.experience_buffer.avgScore, sample_score, sample_game_score, sample_deaths,
    #                       sample_turns))
    #
    #         # create validation games by playing with exploration off
    #         valid_score = 0
    #         valid_game_score = 0
    #         valid_deaths = 0
    #         valid_turns = 0
    #         valid_games = 50
    #         i = 0
    #         while i < valid_games:
    #             game = Game(self.n_players)
    #             score, nTurn, timeSeries = self.play(game, self.time_steps, explore_rate=0)
    #             valid_score += score
    #             valid_game_score += game.score
    #             valid_deaths += Game.MAX_FUSES - game.n_fuse_tokens
    #             valid_turns += nTurn
    #             i += 1
    #         print('{} validation games played'.format(i))
    #
    #         valid_score /= valid_games
    #         valid_game_score /= valid_games
    #         valid_deaths /= valid_games
    #         valid_turns /= valid_games
    #
    #         print('\nvalid score: {}\nvalid game score: {}\nvalid deaths: {}\nvalid turns: {}'
    #               .format(valid_score, valid_game_score, valid_deaths, valid_turns))
    #
    #         # train
    #         loss = self.train(it, 100, checkpoint_folder=save_folder)
    #         print('loss: {}'.format(loss))
    #
    #         # log stuff to stat file
    #         with open(stats_file_name, 'a+') as stats_file:
    #             stats = [it, explore_rate, self.experience_buffer.avgScore,
    #                      sample_discarded, sample_score, sample_game_score, sample_deaths, sample_turns,
    #                      valid_score, valid_game_score, valid_deaths, valid_turns,
    #                      loss]
    #             stats_file.write(','.join(map(str, stats)))
    #             stats_file.write('\n')
    #
    #         # update iteration related variables
    #         explore_rate = max(explore_rate - exploreDecrease, endExploreRate)

    def eval_game(self, game):
        return sum(self.firework_eval[x] for x in game.fireworks) - self.fuse_eval[Game.MAX_FUSES - game.n_fuse_tokens]

    @staticmethod
    def checkpointFile(iteration):
        return 'checkpoint' + str(iteration) + '.ckpt'

    @staticmethod
    def formatBatch(timeSeriesList):
        '''
        Create a batch from a list of time series, each time series is a list of States
        aka a 2D array of State, shape [batch_size x time_steps]
        The resulting batch has each field be a 2D array of shape [batch_size x time_steps],
        where element[i][j] is the value of the field of State at [i][j]
        '''
        return GameState(*zip(*(zip(*s) for s in timeSeriesList)))


# def main():
#     nHiddens = 64
#     batch_size = 128
#     time_steps = 32
#     n_players = 3
#     learn_rate = 1e-5
#
#     trainer = Trainer(3, nHiddens, 2, learn_rate, batch_size, time_steps)
#
#     load_folder = 'save'
#     loadIter = 0
#     if loadIter > 0:
#         trainer.train_model.load_checkpoint(folder=load_folder, filename=Trainer.checkpointFile(loadIter))
#
#     # while True:
#     #     game = Game(n_players)
#     #     trainer.play(game, time_steps, explore_rate=0, show=True)
#     #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     #     input()
#
#     trainer.start(startIteration=loadIter + 1, startExploreRate=0.8, endExploreRate=0.2, exploreDecrease=0.01)

def main():
    n_players = 3
    game_configs = attdict(
        n_players=n_players,
        n_ranks=Game.N_RANKS,
        n_suits=Game.N_SUITS,
        hand_size=Game.HAND_SIZE_PER_N_PLAYERS[n_players],
    )
    model_configs = attdict(
        n_rnn_hiddens=64,
        n_rnn_layers=2,
        n_outputs=Game.ACTIONS_PER_N_PLAYERS[n_players],
        learn_rate=1e-5,
    )
    train_configs = attdict(
        batch_size=128,
        time_steps=32,
    )

    trainer = Trainer(game_configs, model_configs, train_configs)


if __name__ == '__main__':
    main()
