from collections import namedtuple
import os
import tensorflow as tf
from typing import List


class Model(object):
    """PG model that takes in a Hanabi game state and output the probability each action, without any RNN"""

    # STATE_FEATURES = ['cur_player', 'remain_tiles', 'hands', 'hints',
    #                   'n_hint_tokens', 'n_fuse_tokens', 'fire_work', 'last_action', 'valid_mask']

    StateFeatures = namedtuple('StateFeatures', 'remain_tiles hands hints n_hint_tokens n_fuse_tokens '
                                                'fireworks last_action valid_mask')

    def __init__(self, game_configs, model_configs):
        self.model_configs = model_configs
        self.game_configs = game_configs

        # model configs
        n_hiddens = model_configs.n_hiddens
        n_outputs = model_configs.n_outputs
        learn_rate = model_configs.learn_rate
        dropout_rates = model_configs.dropout_rates
        # game configs
        n_ranks = game_configs.n_ranks
        n_suits = game_configs.n_suits
        n_players = game_configs.n_players
        n_tile_types = n_ranks * n_suits
        hand_size = game_configs.hand_size

        self.graph = tf.Graph()
        with self.graph.as_default():
            # -------------------------
            # Input placeholders

            with tf.variable_scope('inputs'):
                self.inputs = namedtuple('InputHeads', 'game_state action reward is_training')(
                    game_state=Model.StateFeatures(
                        remain_tiles=tf.placeholder(tf.float32, shape=[None, n_tile_types], name='remain_tiles'),
                        hands=tf.placeholder(tf.int32, shape=[None, n_players - 1, hand_size], name='hands'),
                        hints=tf.placeholder(tf.float32, shape=[None, n_players, hand_size, n_tile_types],
                                             name='hints'),
                        n_hint_tokens=tf.placeholder(tf.float32, shape=[None], name='n_hint_tokens'),
                        n_fuse_tokens=tf.placeholder(tf.float32, shape=[None], name='n_fuse_tokens'),
                        fireworks=tf.placeholder(tf.int32, shape=[None, n_suits], name='fireworks'),
                        valid_mask=tf.placeholder(tf.float32, shape=[None, n_outputs], name='valid_mask'),
                        last_action=tf.placeholder(tf.int32, shape=[None], name='last_action'),
                    ),
                    # loss_mask=tf.placeholder(tf.int32, shape=[None], name='loss_mask'),
                    action=tf.placeholder(tf.int32, shape=[None], name='action'),
                    reward=tf.placeholder(tf.float32, shape=[None], name='reward'),
                    is_training=tf.placeholder(tf.bool, shape=(), name='is_training')
                )
            # -------------------------
            # Pre-process input heads into 1 large input array:
            batch_size = tf.shape(self.inputs.game_state.n_hint_tokens)[0]

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
            one_hot_last_actions = tf.one_hot(self.inputs.game_state.last_action, n_outputs - 1)

            # Processed input concat into a 2D array (batch_size x num_features)
            concated_inputs = tf.concat(
                [remain_tiles, hands_hints, one_hot_fireworks, n_hint_tokens, n_fuse_tokens,
                 one_hot_last_actions], axis=-1)

            # -------------------------
            # hidden layers (fully connected RELU)
            cur_layer = concated_inputs
            for n_hidden, dropout_rate in zip(n_hiddens, dropout_rates):  # for each hidden layer
                cur_layer = self._create_dense_layer(cur_layer, n_hidden, tf.nn.relu,
                                                     dropout_rate, self.inputs.is_training)

            # -------------------------
            # output layer (probability logit)
            logits = tf.layers.dense(cur_layer, n_outputs)
            unmasked_action_prob = tf.nn.softmax(logits)
            # mask invalid action, then re-normalize to get action probability
            masked_action_prob = unmasked_action_prob * self.inputs.game_state.valid_mask
            self.action_probability = masked_action_prob / tf.reduce_sum(masked_action_prob, axis=1, keepdims=True)

            # -------------------------
            # Back propagation layer
            self.loss = tf.reduce_mean(
                self.inputs.reward * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                    labels=self.inputs.action))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

        # ---------------------------------
        # Initialize graph
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True  # not allow tf to use up all GPU memory, in case of sharing
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))

        # ---------------------------------
        # Save and load operations
        self.saver = tf.train.Saver(self.graph.get_collection('variables'), max_to_keep=10000000)

    @staticmethod
    def _create_dense_layer(input_layer, n_outputs, activation_fn, dropout_rate, is_training):
        dense = tf.layers.dense(input_layer, n_outputs, activation=activation_fn,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        batch_norm = tf.layers.batch_normalization(dense)
        if dropout_rate <= 0:
            return batch_norm
        return tf.layers.dropout(batch_norm, rate=dropout_rate, training=is_training)

    def predict(self, game_states: List[StateFeatures]) -> List[List[float]]:
        """
        Give the predicted Q values of a batch of states
        :param game_states:
        :return:
        """
        game_states = Model._format_input_heads(game_states)
        input_dict = {tensor: value for tensor, value in zip(self.inputs.game_state, game_states)}
        input_dict[self.inputs.is_training] = False
        action_prob = self.sess.run(self.action_probability, feed_dict=input_dict)
        return action_prob

    def train(self, game_states: List[StateFeatures], action: List[int], reward: List[float]) -> float:
        """
        Train the network with the given game and target values
        :param game_states: batch of game states
        :param action: action chosen for each game state
        :param reward: normalized, discounted reward gained after applying action
        :return: average loss
        """
        game_states = Model._format_input_heads(game_states)
        input_dict = {tensor: value for tensor, value in zip(self.inputs.game_state, game_states)}
        input_dict[self.inputs.action] = action
        input_dict[self.inputs.reward] = reward
        input_dict[self.inputs.is_training] = True
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=input_dict)
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
