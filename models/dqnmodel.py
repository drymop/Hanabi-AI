from collections import namedtuple
import numpy as np
import os
import tensorflow as tf
from typing import List

from game import Game


class Model(object):
    """DQN model that takes in a Hanabi game state and output the Q value of each action, without any RNN"""

    # STATE_FEATURES = ['cur_player', 'remain_tiles', 'hands', 'hints',
    #                   'n_hint_tokens', 'n_fuse_tokens', 'fire_work', 'last_action', 'valid_mask']

    StateFeatures = namedtuple('StateFeatures', 'cur_player remain_tiles hands hints n_hint_tokens n_fuse_tokens '
                                                'fireworks last_action valid_mask')

    def __init__(self, game_configs, model_configs):
        self.model_configs = model_configs
        self.game_configs = game_configs

        # game configs
        n_players = game_configs.n_players
        n_actions = Game.ACTIONS_PER_N_PLAYERS[n_players]

        # -------------------------
        # Precomputed neural network's inputs (for extracting states from Game)
        # various valid masks for a game state
        self.valid_mask_none = np.zeros(n_actions)  # none of the actions are valid
        self.valid_mask_do_nothing = np.zeros(n_actions)  # only the last action is valid (do nothing)
        self.valid_mask_do_nothing[-1] = 1

        # -------------------------
        # Define NN graph
        self._define_graph()

        # ---------------------------------
        # Initialize graph
        sess_config = tf.ConfigProto()  # allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True  # not allow tf to use up all GPU memory, in case of sharing
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))

        # ---------------------------------
        # Save and load operations
        self.saver = tf.train.Saver(self.graph.get_collection('variables'), max_to_keep=10000000)

    def _define_graph(self):
        model_configs = self.model_configs
        game_configs = self.game_configs
        # model configs
        n_hiddens = model_configs.n_hiddens
        learn_rate = model_configs.learn_rate
        dropout_rates = model_configs.dropout_rates
        # game configs
        n_ranks = game_configs.n_ranks
        n_suits = game_configs.n_suits
        n_players = game_configs.n_players
        n_tile_types = n_ranks * n_suits
        hand_size = game_configs.hand_size
        n_actions = Game.ACTIONS_PER_N_PLAYERS[n_players]
        self.graph = tf.Graph()
        with self.graph.as_default():
            # -------------------------
            # Input placeholders
            with tf.variable_scope('inputs'):
                self.inputs = namedtuple('InputHeads', 'game_state loss_mask targets is_training')(
                    game_state=Model.StateFeatures(
                        cur_player=tf.placeholder(tf.int32, shape=[None], name='cur_player'),
                        remain_tiles=tf.placeholder(tf.float32, shape=[None, n_tile_types], name='remain_tiles'),
                        hands=tf.placeholder(tf.int32, shape=[None, n_players - 1, hand_size], name='hands'),
                        hints=tf.placeholder(tf.float32, shape=[None, n_players, hand_size, n_tile_types],
                                             name='hints'),
                        n_hint_tokens=tf.placeholder(tf.float32, shape=[None], name='n_hint_tokens'),
                        n_fuse_tokens=tf.placeholder(tf.float32, shape=[None], name='n_fuse_tokens'),
                        fireworks=tf.placeholder(tf.int32, shape=[None, n_suits], name='fireworks'),
                        valid_mask=tf.placeholder(tf.float32, shape=[None, n_actions], name='valid_mask'),
                        last_action=tf.placeholder(tf.int32, shape=[None], name='last_action'),
                    ),
                    loss_mask=tf.placeholder(tf.int32, shape=[None], name='loss_mask'),
                    targets=tf.placeholder(tf.float32, shape=[None, n_actions], name='targets'),
                    is_training=tf.placeholder(tf.bool, shape=(), name='is_training')
                )
            # -------------------------
            # Pre-process input heads into 1 large input array:
            batch_size = tf.shape(self.inputs.game_state.cur_player)[0]

            # Create one_hot vectors for current player's number
            one_hot_cur_player = tf.one_hot(self.inputs.game_state.cur_player, n_players)

            # Create one_hot vectors for hands
            one_hot_hands = tf.one_hot(self.inputs.game_state.hands, n_tile_types)

            # flatten hands and hints
            hands_hints = tf.concat([one_hot_hands, self.inputs.game_state.hints], axis=1)
            shape = hands_hints.shape
            hands_hints = tf.reshape(hands_hints, [-1, shape[1] * shape[2] * shape[3]])  # flatten

            # subtract the seen tiles from other players
            remain_tiles = self.inputs.game_state.remain_tiles - tf.reduce_sum(one_hot_hands, axis=[1, 2])

            # reshape n_hint_tokens and n_fuse_tokens from float to array of 1 element
            n_hint_tokens = tf.reshape(self.inputs.game_state.n_hint_tokens, [batch_size, 1])
            n_fuse_tokens = tf.reshape(self.inputs.game_state.n_fuse_tokens, [batch_size, 1])
            # shape: [batch_size, n_ranks, n_suits]
            one_hot_fireworks = tf.one_hot(self.inputs.game_state.fireworks, n_ranks, axis=1)
            one_hot_fireworks = tf.reshape(one_hot_fireworks, [batch_size, n_tile_types])

            # one hot vector for last action
            one_hot_last_actions = tf.one_hot(self.inputs.game_state.last_action, n_actions - 1)

            # Processed input concat into a 2D array (batch_size x num_features)
            concated_inputs = tf.concat(
                [one_hot_cur_player, remain_tiles, hands_hints, one_hot_fireworks, n_hint_tokens, n_fuse_tokens,
                 one_hot_last_actions], axis=-1)

            # -------------------------
            # hidden layers (fully connected RELU)
            cur_layer = concated_inputs
            for n_hidden, dropout_rate in zip(n_hiddens, dropout_rates):  # for each hidden layer
                cur_layer = self._create_dense_layer(cur_layer, n_hidden, tf.nn.relu,
                                                     dropout_rate, self.inputs.is_training)

            # -------------------------
            # output layer (fully connected, linear activation), masked by valid_mask input
            cur_layer = tf.layers.dense(cur_layer, n_actions)
            self.outputs = tf.multiply(cur_layer, self.inputs.game_state.valid_mask)

            # -------------------------
            # Back propagation layer
            one_hot_loss_mask = tf.one_hot(self.inputs.loss_mask, n_actions)

            self.loss = tf.losses.mean_squared_error(self.inputs.targets,
                                                     self.outputs,
                                                     weights=one_hot_loss_mask)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

    @staticmethod
    def _create_dense_layer(input_layer, n_outputs, activation_fn, dropout_rate, is_training):
        dense = tf.layers.dense(input_layer, n_outputs, activation=activation_fn,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        batch_norm = tf.layers.batch_normalization(dense)
        if dropout_rate <= 0:
            return batch_norm
        return tf.layers.dropout(batch_norm, rate=dropout_rate, training=is_training)

    @staticmethod
    def _encode_hints(hint):
        return np.fromiter((1 if b1 and b2 else 0 for b1 in hint[1] for b2 in hint[0]),
                           dtype=np.int8, count=Game.N_TYPES)

    def extract_features(self, game: Game, last_action: int = -1):
        """
        Create n states (n = n_players), each is a game state from each player's point of view
        """
        n_players = game.n_players
        remain_tiles = np.fromiter(game.n_tiles_per_type, dtype=np.int8, count=len(game.n_tiles_per_type))
        # all_hints[player]
        all_hands = [np.fromiter((tile.id for tile in hand), dtype=np.int8, count=game.hand_size)
                     for hand in game.hands]
        # all_hints[player][tile]
        all_hints = [[Model._encode_hints(tile_hint) for tile_hint in player_hint] for player_hint in game.hints]
        fireworks = np.fromiter(game.fireworks, dtype=np.int8, count=len(game.fireworks))

        if game.is_over:
            valid_mask_cur_player = self.valid_mask_none
            valid_mask_other_player = self.valid_mask_none
        else:
            valid_mask_cur_player = np.fromiter((1 if x else 0 for x in game.is_valid_action),
                                                dtype=np.int8, count=game.n_actions)
            valid_mask_other_player = self.valid_mask_do_nothing

        game_states = [None] * n_players  # type: List[Model.StateFeatures]
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

    def predict(self, game_states: List[StateFeatures]) -> List[List[float]]:
        """
        Give the predicted Q values of a batch of states
        :param game_states:
        :return:
        """
        game_states = Model._format_input_heads(game_states)
        input_dict = {tensor: value for tensor, value in zip(self.inputs.game_state, game_states)}
        input_dict[self.inputs.is_training] = False
        q_values = self.sess.run(self.outputs, feed_dict=input_dict)
        return q_values

    def train(self, game_states: List[StateFeatures], targets: List[List[float]], loss_mask: List[int]) -> float:
        """
        Train the network with the given game and target values
        :param game_states: batch of game states
        :param targets: the corrected Q values for each state in the batch
        :param loss_mask: the action whose Q value needs to be learned for each state in batch
        :return: average loss
        """
        game_states = Model._format_input_heads(game_states)
        input_dict = {tensor: value for tensor, value in zip(self.inputs.game_state, game_states)}
        input_dict[self.inputs.loss_mask] = loss_mask
        input_dict[self.inputs.targets] = targets
        input_dict[self.inputs.is_training] = True
        loss, _ = self.sess.run([self.loss, self.train_step], feed_dict=input_dict)
        return loss

    @staticmethod
    def _format_input_heads(game_states: List[StateFeatures]) -> StateFeatures:
        """
        :param game_states: batch of game states
        :return: StateFeatures where each feature is the combined feature of the batch
        """
        return Model.StateFeatures(*zip(*game_states))

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.ckpt'):
        file_path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        with self.graph.as_default():
            self.saver.save(self.sess, file_path)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.ckpt'):
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path + '.meta'):
            raise FileNotFoundError('No model in path {}'.format(file_path))
        with self.graph.as_default():
            self.saver.restore(self.sess, file_path)
