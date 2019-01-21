from collections import namedtuple
import numpy as np
import os
import random
import tensorflow as tf
from pprint import pprint as pp

from game import Game
from models.DQNModel2ver1 import Model
from utils.ConsoleDisplay import displayState, displayAction
from utils.ExperienceBuffer import Experience, ExperienceBuffer

GameState = namedtuple('GameState', ' '.join(['cur_player', 'remain_tiles', 'hands', 'hints', 'hint_tokens', 'fuse_tokens', 'fireworks', 'action_and_reward', 'valid_mask', 'loss_mask']))

class Trainer:

    def __init__(self, nPlayers, nHiddens, nLayers, learn_rate, batchSize, timeSteps):
        self.nPlayers = nPlayers
        self.batchSize = batchSize
        self.timeSteps = timeSteps

        # models to train
        nActions = Game.ACTIONS_PER_N_PLAYERS[nPlayers]
        handSize = Game.HAND_SIZE_PER_N_PLAYERS[nPlayers]
        self.trainModel = Model(nHiddens, nLayers, nActions, learn_rate=learn_rate)
        self.targetModel = Model(nHiddens, nLayers, nActions, learn_rate=learn_rate)

        # experience relay buffer
        self.bufferSize = 40000
        self.experienceBuffer = ExperienceBuffer(self.bufferSize)

        # scoring
        self.fireworkEvals = [0, 1, 2.2, 3.6, 5.3, 7.3]
        fuse_val = 0.3
        fuse_scale_factor = 1.1
        self.fuseEvals = [0, fuse_val]
        for i in range(35):
            fuse_val *= fuse_scale_factor
            self.fuseEvals.append(self.fuseEvals[-1] + fuse_val)

        # zero rnn state for a game
        self.gameZeroState = np.zeros((nLayers, 2, nPlayers, nHiddens))
        self.batchZeroState = np.zeros((nLayers, 2, batchSize, nHiddens))

        # various loss masks for a state
        self.maskOnes = np.ones(nActions)
        self.maskZeros = np.zeros(nActions)
        # valid action masks
        self.maskNonCurPlayer = np.zeros(nActions)
        self.maskNonCurPlayer[-1] = 1

    def onehotHint(self, hint):
        return [1 if b1 and b2 else 0 for b1 in hint[1] for b2 in hint[0]]

    def createGameStates(self, g, last_action=-1):
        '''
        Create n states (n = g.n_players), each is a game state from each player's point of view
        '''
        cur_player = [(g.curPlayer-player)%g.nPlayers for player in range(g.nPlayers)]

        remain_tiles = [g.nRemainTiles.copy()] * g.nPlayers

        all_hands = [[tile.id for tile in hand] for hand in g.hands]
        hands = [[all_hands[(player+i) % g.nPlayers] for i in range(1, g.nPlayers)] for player in range(g.nPlayers)]
        
        onehotHints = [[self.onehotHint(tile) for tile in hint] for hint in g.hints]
        hints = [[onehotHints[(player+i) % g.nPlayers] for i in range(g.nPlayers)] for player in range(g.nPlayers)]

        hint_tokens = [g.nHintTokens] * g.nPlayers
        fuse_tokens = [g.nFuseTokens] * g.nPlayers
        fireworks = [g.fireworks.copy()] * g.nPlayers
        action_and_rewards = [[-1, 0] for _ in range(g.nPlayers)]
        
        if g.is_over():
            valid_masks = [self.maskZeros] * g.nPlayers
            loss_masks = valid_masks
        else:
            valid_masks = [g.getValidActions() if p == g.curPlayer else self.maskNonCurPlayer for p in range(g.nPlayers)]
            #loss_masks = [self.maskOnes if p == g.cur_player else self.maskZeros for p in range(g.n_players)]
            loss_masks = [self.maskOnes] * g.nPlayers

        states = [GameState(*t) for t in zip(cur_player, remain_tiles, hands, hints, hint_tokens, fuse_tokens, fireworks, action_and_rewards, valid_masks, loss_masks)]

        return states

    def getTrainBatch(self, batchSize, timeSteps):
        batch = self.experienceBuffer.sample(batchSize)
        for i in range(batchSize):
            # trim the series to timeSteps steps
            series = batch[i].experience
            r = random.randrange(len(series)-timeSteps)
            batch[i] = series[r:r+timeSteps+1]
        return batch


    def play(self, game, minLength, explore_rate=0.1, show=False):
        # 2d array of state, recording the time series of each player
        timeSeries = [[] for _ in range(game.nPlayers)]

        rnnStateTuple = self.gameZeroState
        score = self.eval(game)
        while not game.is_over():

            gStates = self.createGameStates(game)
            for player in range(game.nPlayers):
                timeSeries[player].append(gStates[player])

            x = Trainer.formatBatch([s] for s in gStates)

            batchQValues, rnnStateTuple = self.trainModel.predict(x, rnnStateTuple)

            # get the q-prediction of the current player at the only time step
            qValues = batchQValues[game.curPlayer][-1]
            
            validActions = game.getValidActions()

            if random.random() > explore_rate:
                # choose action with max q value
                m = max(qValues[i] for i in range(len(qValues)) if validActions[i])
                choices = [i for i in range(len(qValues)) if qValues[i] == m]
            else:
                # choose action randomly
                choices = [i for i, b in enumerate(validActions) if b]
            actionInd = random.choice(choices)
            action = game.actions[actionInd]

            # save the action
            gStates[game.curPlayer].action_and_reward[0] = actionInd

            if show:
                displayState(game, firstPerson=False)
                pp(batchQValues)

                print('\nAction: {}'.format(actionInd))
                displayAction(game, action)
                print()
                input('Press ENTER...')


            game.play(action)

            # get reward
            newScore = self.eval(game)
            for state in gStates:
                state.action_and_reward[1] = newScore - score
            score = newScore
            

        if show:
            displayState(game, False)
            print(game.score, end=" ")
            print(game.nFuseTokens)

        # Add the terminal state, as well as pad to minimum required length
        nTurn = len(timeSeries[0])
        pad = max(0, minLength - nTurn) + 1
        terminalStates = self.createGameStates(game)
        for _ in range(pad):
            for player in range(game.nPlayers):
                    timeSeries[player].append(terminalStates[player])

        return score, nTurn, timeSeries

    def play_batch(self, batch_size, min_length, explore_rate=0.1):
        # 2d array of state, recording the time series of each player
        timeSeries = [[] for _ in range(game.nPlayers)]

        rnnStateTuple = self.gameZeroState
        score = self.eval(game)
        while not game.is_over():

            gStates = self.createGameStates(game)
            for player in range(game.nPlayers):
                timeSeries[player].append(gStates[player])

            x = Trainer.formatBatch([s] for s in gStates)

            batchQValues, rnnStateTuple = self.trainModel.predict(x, rnnStateTuple)

            # get the q-prediction of the current player at the only time step
            qValues = batchQValues[game.curPlayer][-1]
            
            validActions = game.getValidActions()

            if random.random() > explore_rate:
                # choose action with max q value
                m = max(qValues[i] for i in range(len(qValues)) if validActions[i])
                choices = [i for i in range(len(qValues)) if qValues[i] == m]
            else:
                # choose action randomly
                choices = [i for i, b in enumerate(validActions) if b]
            actionInd = random.choice(choices)
            action = game.actions[actionInd]

            # save the action
            gStates[game.curPlayer].action_and_reward[0] = actionInd

            if show:
                displayState(game, firstPerson=False)
                pp(batchQValues)

                print('\nAction: {}'.format(actionInd))
                displayAction(game, action)
                print()
                input('Press ENTER...')


            game.play(action)

            # get reward
            newScore = self.eval(game)
            for state in gStates:
                state.action_and_reward[1] = newScore - score
            score = newScore
            

        if show:
            displayState(game, False)
            print(game.score, end=" ")
            print(game.nFuseTokens)

        # Add the terminal state, as well as pad to minimum required length
        nTurn = len(timeSeries[0])
        pad = max(0, minLength - nTurn) + 1
        terminalStates = self.createGameStates(game)
        for _ in range(pad):
            for player in range(game.nPlayers):
                    timeSeries[player].append(terminalStates[player])

        return score, nTurn, timeSeries

    def playRandom(self, game, minLength, show=False):
        """Play a game randomly and return the episode.

        Faster than calling play with explore_rate=1, as no neural network is involved.
        """

        # 2d array of state, recording the time series of each player
        timeSeries = [[] for _ in range(game.nPlayers)]

        #rnnStateTuple = self.gameZeroState
        score = self.eval(game)
        while not game.is_over():
            gStates = self.createGameStates(game)
            for player in range(game.nPlayers):
                timeSeries[player].append(gStates[player])

            validActions = game.getValidActions()
            # choose action randomly
            choices = [i for i, b in enumerate(validActions) if b]
            actionInd = random.choice(choices)
            action = game.actions[actionInd]

            # save the action
            gStates[game.curPlayer].action_and_reward[0] = actionInd

            if show:
                display(game, False)
                pp(batchQValues)

                print('\nAction: {}'.format(actionInd))
                pp(action.__dict__)
                print()
                input()


            game.play(action)

            # get reward
            newScore = self.eval(game)
            reward = newScore - score
            score = newScore
            for state in gStates:
                state.action_and_reward[1] = reward

        # display final state
        if show:
            displayState(game, False)
            print(game.score, end=" ")
            print(game.nFuseTokens)

        # Add the terminal state, as well as pad to minimum required length
        nTurn = len(timeSeries[0])
        pad = max(0, minLength - nTurn) + 1
        terminalStates = self.createGameStates(game)
        for _ in range(pad):
            for player in range(game.nPlayers):
                    timeSeries[player].append(terminalStates[player])

        return score, nTurn, timeSeries


    def train(self, iteration, epoch, checkpoint_folder='save', discount_rate=0.99):
        saveFile = Trainer.checkpointFile(iteration)
        self.trainModel.save_checkpoint(folder=checkpoint_folder, filename=saveFile)
        self.targetModel.load_checkpoint(folder=checkpoint_folder, filename=saveFile)

        # train for some epochs
        avgLoss = 0
        for epochI in range(epoch):
            # print('EPOCH ::: ' + str(epochI + 1))

            # get experiences
            batch = self.getTrainBatch(self.batchSize, self.timeSteps)

            # Q value for such experience
            targetQ, _ = self.targetModel.predict(Trainer.formatBatch(batch), self.batchZeroState)

            # train agains the target
            for i in range(self.batchSize):
                for j in range(self.timeSteps):
                    actionTaken, reward = batch[i][j].action_and_reward
                    targetQ[i][j][actionTaken] = reward + discount_rate * max(targetQ[i][j+1])

            # remove the last time step
            for series in batch:
                series.pop()
            targetQ = np.delete(targetQ, -1, 1)
            t = Trainer.formatBatch(batch)

            loss = self.trainModel.train(Trainer.formatBatch(batch), self.batchZeroState, targetQ)
            print(loss)
            avgLoss += loss
        return avgLoss / epoch

    def start(self, startIteration=0, startExploreRate=1, endExploreRate=0.05, exploreDecrease=0.015):
        # fill up the buffer
        print('==================================== FILLING BUFFER ===================================')
        avgScore = 0
        avgTurn = 0
        # n_random_games = self.bufferSize//self.n_players//2
        n_random_games = 1
        i = 0
        while i < n_random_games:
            game = Game(self.nPlayers)
            score, nTurn, timeSeries = self.playRandom(game, self.timeSteps)

            i += 1
            for s in timeSeries:
                self.experienceBuffer.add(Experience(score, s), weighted=True)
            avgScore += score
            avgTurn += nTurn
            if (i % 200) == 0:
                print('{} games played'.format(i))
                print('buffer score: {}'.format(self.experienceBuffer.avgScore))
                print('score: {}'.format(avgScore / i))
                print('turn : {}'.format(avgTurn / i))
        print('score: {}'.format(avgScore / n_random_games))
        print('turn : {}'.format(avgTurn / n_random_games))

        explore_rate = max(startExploreRate-(startIteration-1)*exploreDecrease, endExploreRate)

        # create folder and files for recording train progress
        save_folder = 'save'
        stats_file_name = 'save/stats.csv'
        if not os.path.exists(save_folder):
            print("Save directory does not exist! Making directory {}".format(save_folder))
            os.mkdir(save_folder)

        with open(stats_file_name, 'a+') as stats_file:
            stats_file.write('iter, explore_rate, buffer_score, sample_discarded, sample_score, sample_game_score, sample_deaths, sample_turns, validation_score, validation_game_score, validation_deaths, validation_turns, loss\n')

        # start training
        for it in range(startIteration, 100000+startIteration):
            print('========================================= ITER {} ==================================='.format(it))
            print('explore rate: {}'.format(explore_rate))

            # create sample games by playing with exploration on
            sample_discarded = 0
            sample_score = 0
            sample_game_score = 0
            sample_deaths = 0
            sample_turns = 0
            sample_games = 500
            i = 0
            while i < sample_games:
                game = Game(self.nPlayers)
                score, nTurn, timeSeries = self.play(game, self.timeSteps, explore_rate=explore_rate)
                if game.score <= 0:
                    sample_discarded += 1
                    continue
                for s in timeSeries:
                    self.experienceBuffer.add(Experience(score, s), weighted=True)
                sample_score += score
                sample_game_score += game.score
                sample_deaths += Game.MAX_FUSES - game.n_fuse_tokens
                sample_turns += nTurn
                if (i % 100) == 0:
                    print('{} sample games played, discarded {}'.format(i, sample_discarded))
                i += 1

            sample_score /= sample_games
            sample_game_score /= sample_games
            sample_deaths /= sample_games
            sample_turns /= sample_games
            
            print('buffer score: {}\nsample score: {}\nsample game score: {}\nsample deaths: {}\nsample turns: {}'
                .format(self.experienceBuffer.avgScore, sample_score, sample_game_score, sample_deaths, sample_turns))

            # create validation games by playing with exploration off
            valid_score = 0
            valid_game_score = 0
            valid_deaths = 0
            valid_turns = 0
            valid_games = 50
            i = 0
            while i < valid_games:
                game = Game(self.nPlayers)
                score, nTurn, timeSeries = self.play(game, self.timeSteps, explore_rate=0)
                valid_score += score
                valid_game_score += game.score
                valid_deaths += Game.MAX_FUSES - game.n_fuse_tokens
                valid_turns += nTurn
                i += 1
            print('{} validation games played'.format(i))

            valid_score /= valid_games
            valid_game_score /= valid_games
            valid_deaths /= valid_games
            valid_turns /= valid_games
            
            print('\nvalid score: {}\nvalid game score: {}\nvalid deaths: {}\nvalid turns: {}'
                .format(valid_score, valid_game_score, valid_deaths, valid_turns))

            # train
            loss = self.train(it, 100, checkpoint_folder=save_folder)
            print('loss: {}'.format(loss))

            # log stuff to stat file
            with open(stats_file_name, 'a+') as stats_file:
                stats = [it, explore_rate, self.experienceBuffer.avgScore, 
                        sample_discarded, sample_score, sample_game_score, sample_deaths, sample_turns, 
                        valid_score, valid_game_score, valid_deaths, valid_turns,
                        loss]
                stats_file.write(','.join(map(str, stats)))
                stats_file.write('\n')

            # update iteration related variables
            explore_rate = max(explore_rate - exploreDecrease, endExploreRate)


    def eval(self, game):
        return sum(self.fireworkEvals[x] for x in game.fireworks) - self.fuseEvals[Game.MAX_FUSES - game.nFuseTokens]

    @staticmethod
    def checkpointFile(iteration):
        return 'checkpoint' + str(iteration) + '.ckpt'


    @staticmethod
    def formatBatch(timeSeriesList):
        '''
        Create a batch from a list of time series, each time series is a list of States
        aka a 2D array of State, shape [batchSize x timeSteps]
        Time series must have equals length = timeSteps
        The resulting batch has each field be a 2D array of shape [batchSize x timeSteps],
        where element[i][j] is the value of the field of State at [i][j]
        '''
        return GameState(*zip(*(zip(*s) for s in timeSeriesList)))



nHiddens = 64
batchSize = 128
timeSteps = 32
nPlayers = 3
learn_rate = 1e-5

trainer = Trainer(3, nHiddens, 2, learn_rate, batchSize, timeSteps)

load_folder = 'save'
loadIter = 0
if loadIter > 0:
    trainer.trainModel.load_checkpoint(folder=load_folder, filename=Trainer.checkpointFile(loadIter))

# while True:
#     game = Game(n_players)
#     trainer.play(game, timeSteps, explore_rate=0, show=True)
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     input()

trainer.start(startIteration=loadIter+1, startExploreRate=0.8, endExploreRate=0.2, exploreDecrease=0.01)