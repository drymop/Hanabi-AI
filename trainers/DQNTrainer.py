from collections import namedtuple
import numpy as np
import random
import tensorflow as tf
from pprint import pprint as pp

from StandardGame import Game
from models.DQNModel2 import Model
from utils.ConsoleDisplay import displayState, displayAction
from utils.ExperienceBuffer import Experience, ExperienceBuffer

iterations = 100
game_per_iteration = 50
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 0.1
nnet_hidden = 32

GameState = namedtuple('GameState', ' '.join(['cur_player', 'remain_tiles', 'hands', 'hints', 'hint_tokens', 'fuse_tokens', 'fireworks', 'action_and_reward', 'valid_mask', 'loss_mask']))

class Trainer:

    def __init__(self, nPlayers, nHiddens, batchSize, timeSteps):
        self.nPlayers = nPlayers
        self.batchSize = batchSize
        self.timeSteps = timeSteps

        # models to train
        nActions = Game.ACTIONS_PER_N_PLAYERS[nPlayers]
        handSize = Game.HAND_SIZE_PER_N_PLAYERS[nPlayers]
        self.trainModel = Model(nHiddens, nActions, nPlayers, handSize)
        self.targetModel = Model(nHiddens, nActions, nPlayers, handSize)

        # experience relay buffer
        self.bufferSize = 45000
        self.experienceBuffer = ExperienceBuffer(self.bufferSize)

        # precompute onehot vectors
        self.onehotPlayers = np.identity(nPlayers)
        self.onehotTiles = np.identity(Game.N_TYPES)

        # zero rnn state for a game
        game_zero_state_tensor = self.trainModel.rnn_cell.zero_state(nPlayers, tf.float32)
        batch_zero_state_tensor = self.trainModel.rnn_cell.zero_state(batchSize, tf.float32)

        self.gameZeroState = np.zeros((2, nPlayers, nHiddens))
        self.batchZeroState = np.zeros((2, batchSize, nHiddens))

        # various loss masks for a state
        self.maskOnes = np.ones(nActions)
        self.maskZeros = np.zeros(nActions)
        # valid action masks
        self.maskNonCurPlayer = np.zeros(nActions)
        self.maskNonCurPlayer[-1] = 1

    def onehotHint(self, hint):
        return [1 if b1 and b2 else 0 for b1 in hint[1] for b2 in hint[0]]

    def createGameStates(self, g):
        '''
        Create n states (n = g.nPlayers), each is a game state from each player's point of view
        '''
        cur_player = [self.onehotPlayers[(g.curPlayer-player)%g.nPlayers] for player in range(g.nPlayers)]

        remain_tiles = [g.nRemainTiles] * g.nPlayers

        onehotHands = [[self.onehotTiles[tile.id] for tile in hand] for hand in g.hands]
        hands = [[onehotHands[(player+i) % g.nPlayers] for i in range(1, g.nPlayers)] for player in range(g.nPlayers)]
        
        onehotHints = [[self.onehotHint(tile) for tile in hint] for hint in g.hints]
        hints = [[onehotHints[(player+i) % g.nPlayers] for i in range(g.nPlayers)] for player in range(g.nPlayers)]

        hint_tokens = [g.nHintTokens] * g.nPlayers
        fuse_tokens = [g.nFuseTokens] * g.nPlayers
        fireworks = [g.fireworks] * g.nPlayers
        action_and_rewards = [[-1, None] for _ in range(g.nPlayers)]
        
        if g.isOver():
            valid_masks = [self.maskZeros] * g.nPlayers
            loss_masks = valid_masks
        else:
            valid_masks = [g.getValidActions() if p == g.curPlayer else self.maskNonCurPlayer for p in range(g.nPlayers)]
            loss_masks = [self.maskOnes if p == g.curPlayer else self.maskZeros for p in range(g.nPlayers)]

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


    #nnet = Model(nnet_hidden, n_actions)
    def play(self, game, minLength, exploration_rate=0.1, show=False):
        # 2d array of state, recording the time series of each player
        timeSeries = [[] for _ in range(game.nPlayers)]

        rnnStateTuple = self.gameZeroState
        score = Trainer.getScore(game)
        while not game.isOver():

            gStates = self.createGameStates(game)
            for player in range(game.nPlayers):
                timeSeries[player].append(gStates[player])

            x = Trainer.formatBatch([s] for s in gStates)

            batchQValues, rnnStateTuple = self.trainModel.predict(x, rnnStateTuple)
            pp(rnnStateTuple)
            input()

            # get the q-prediction of the current player at the only time step
            qValues = batchQValues[game.curPlayer][-1]
            
            validActions = game.getValidActions()

            if random.random() > exploration_rate:
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
                input()


            game.play(action)

            # get reward
            newScore = Trainer.getScore(game)
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
        # 2d array of state, recording the time series of each player
        timeSeries = [[] for _ in range(game.nPlayers)]

        #rnnStateTuple = self.gameZeroState
        score = Trainer.getScore(game)
        while not game.isOver():
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
            newScore = Trainer.getScore(game)
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


    def train(self, iteration, epoch, discount_rate=0.99):
        saveFile = Trainer.checkpointFile(iteration)
        self.trainModel.save_checkpoint(folder='save', filename=saveFile)
        self.targetModel.load_checkpoint(folder='save', filename=saveFile)

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
            avgLoss += loss
        return avgLoss / epoch

    def start(self, startIteration=0, startExploreRate=1, endExploreRate=0.05, exploreDecrease=0.015):
        exploration_rate = 1
        # fill up the buffer
        # print('==================================== FILLING BUFFER ===================================')
        # avgScore = 0
        # avgTurn = 0
        # nGames = self.bufferSize//self.nPlayers//2
        # i = 0
        # while i < nGames:
        #     game = Game(self.nPlayers)
        #     score, nTurn, timeSeries = self.playRandom(game, self.timeSteps)

        #     i += 1
        #     for s in timeSeries:
        #         self.experienceBuffer.add(Experience(score, s), True)
        #     avgScore += score
        #     avgTurn += nTurn
        #     if (i % 200) == 0:
        #         print('{} games played'.format(i))
        #         print('buffer score: {}'.format(self.experienceBuffer.avgScore))
        #         print('score: {}'.format(avgScore / i))
        #         print('turn : {}'.format(avgTurn / i))
        # print('score: {}'.format(avgScore / nGames))
        # print('turn : {}'.format(avgTurn / nGames))


        # start training
        for it in range(startIteration, 100000+startIteration):
            print('========================================= ITER {} ==================================='.format(it))
            avgScore = 0
            avgTurn = 0
            nGames = 1000
            for i in range(nGames):
                game = Game(self.nPlayers)
                score, nTurn, timeSeries = self.play(game, self.timeSteps, exploration_rate=exploration_rate)
                for s in timeSeries:
                    self.experienceBuffer.add(Experience(score, s), True)
                avgScore += score
                avgTurn += nTurn
                if (i % 100) == 0:
                    print('{} games played'.format(i))
            print('buffer score: {}'.format(self.experienceBuffer.avgScore))
            print('score: {}'.format(avgScore / nGames))
            print('turn : {}'.format(avgTurn / nGames))

            loss = self.train(it, 100)
            print('loss: {}'.format(loss))

            exploration_rate = max(exploration_rate - exploreDecrease, endExploreRate)



    @staticmethod
    def getScore(game):
        return game.score + 0.5 * game.nFuseTokens

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



nHiddens = 128
batchSize = 128
timeSteps = 32
nPlayers = 3

trainer = Trainer(3, nHiddens, batchSize, timeSteps)

#trainer.trainModel.load_checkpoint(folder='dqn_save_2', filename=Trainer.checkpointFile(151))
# while True:
#     game = Game(nPlayers)
#     trainer.play(game, timeSteps, exploration_rate=0.1, show=True)
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     input()

pp(trainer.gameZeroState)
pp(trainer.batchZeroState)

input()
trainer.start(startIteration=0, startExploreRate=0.8, endExploreRate=0.05, exploreDecrease=0.02)

