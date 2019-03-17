from collections import namedtuple
import os
import tensorflow as tf


class Model(object):
    """DQN model that takes in a sequence of Hanabi game states and output the Q value of each action"""

    # STATE_FEATURES = ['cur_player', 'remain_tiles', 'hands', 'hints',
    #                   'n_hint_tokens', 'n_fuse_tokens', 'fire_work', 'last_action', 'valid_mask']

    StateFeatures = namedtuple('StateFeatures', 'cur_player remain_tiles hands hints n_hint_tokens n_fuse_tokens '
                                                'fireworks last_action valid_mask')

    def __init__(self, game_configs, model_configs):
        self.model_configs = model_configs
        self.game_configs = game_configs

        # model configs
        n_rnn_hiddens = model_configs.n_rnn_hiddens
        n_rnn_layers = model_configs.n_rnn_layers
        n_dense_after_rnn = model_configs.n_dense_after_rnn
        n_dense_before_rnn = model_configs.n_dense_before_rnn
        n_outputs = model_configs.n_outputs
        learn_rate = model_configs.learn_rate
        dropout_rate = model_configs.dropout_rate
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
                self.inputs = namedtuple('InputHeads', 'game_state rnn_init_state loss_mask targets is_training')(
                    game_state=Model.StateFeatures(
                        cur_player=tf.placeholder(tf.int32, shape=[None, None], name='cur_player'),
                        remain_tiles=tf.placeholder(tf.float32, shape=[None, None, n_tile_types], name='remain_tiles'),
                        hands=tf.placeholder(tf.int32, shape=[None, None, n_players - 1, hand_size], name='hands'),
                        hints=tf.placeholder(tf.float32, shape=[None, None, n_players, hand_size, n_tile_types],
                                             name='hints'),
                        n_hint_tokens=tf.placeholder(tf.float32, shape=[None, None], name='n_hint_tokens'),
                        n_fuse_tokens=tf.placeholder(tf.float32, shape=[None, None], name='n_fuse_tokens'),
                        fireworks=tf.placeholder(tf.int32, shape=[None, None, n_suits], name='fireworks'),
                        valid_mask=tf.placeholder(tf.float32, shape=[None, None, n_outputs], name='valid_mask'),
                        last_action=tf.placeholder(tf.int32, shape=[None, None], name='last_action'),
                    ),
                    rnn_init_state=tf.placeholder(tf.float32,
                                                  shape=[n_rnn_layers, 2, None, n_rnn_hiddens],
                                                  name='rnn_init_state'),
                    loss_mask=tf.placeholder(tf.int32, shape=[None, None], name='loss_mask'),
                    targets=tf.placeholder(tf.float32, shape=[None, None, n_outputs], name='targets'),
                    is_training=tf.placeholder(tf.bool, shape=(), name='is_training')
                )
            # -------------------------
            # Pre-process input heads into 1 large input array:
            shape = tf.shape(self.inputs.game_state.cur_player)
            batch_size = shape[0]
            time_steps = shape[1]

            # Create one_hot vectors for current player's number
            one_hot_cur_player = tf.one_hot(self.inputs.game_state.cur_player, n_players)

            # Create one_hot vectors for hands
            one_hot_hands = tf.one_hot(self.inputs.game_state.hands, n_tile_types)

            # flatten hands and hints
            hands_hints = tf.concat([one_hot_hands, self.inputs.game_state.hints], axis=2)
            shape = hands_hints.shape
            hands_hints = tf.reshape(hands_hints, [-1, time_steps, shape[2] * shape[3] * shape[4]])  # flatten

            # subtract the seen tiles from other players
            remain_tiles = self.inputs.game_state.remain_tiles - tf.reduce_sum(one_hot_hands, axis=[2, 3])

            # normalize integral quantities into (0, 1) range
            # remain_tiles = tf.scalar_mul(tf.constant(1/MAX_N_PER_TYPES), remain_tiles)
            # n_hint_tokens  = tf.scalar_mul(tf.constant(1/MAX_HINT_TOKENS), self.inputs.game_state.n_hint_tokens)
            # n_fuse_tokens  = tf.scalar_mul(tf.constant(1/MAX_FUSE_TOKENS), self.inputs.game_state.n_fuse_tokens)

            # reshape n_hint_tokens and n_fuse_tokens from float to array of 1 element
            n_hint_tokens = tf.reshape(self.inputs.game_state.n_hint_tokens, [batch_size, time_steps, 1])
            n_fuse_tokens = tf.reshape(self.inputs.game_state.n_fuse_tokens, [batch_size, time_steps, 1])
            # shape: [batch_size, time_steps, n_ranks, n_suits]
            one_hot_fireworks = tf.one_hot(self.inputs.game_state.fireworks, n_ranks, axis=2)
            one_hot_fireworks = tf.reshape(one_hot_fireworks, [batch_size, time_steps, n_tile_types])

            # one hot vector for last action
            one_hot_last_actions = tf.one_hot(self.inputs.game_state.last_action, n_outputs - 1)

            # Processed input concat into a 2D array (batch_size x num_features)
            concated_inputs = tf.concat(
                [one_hot_cur_player, remain_tiles, hands_hints, one_hot_fireworks, n_hint_tokens, n_fuse_tokens,
                 one_hot_last_actions], axis=-1)

            # -------------------------
            # compress the sparse input through dense layers
            layer = concated_inputs
            for i in range(n_dense_before_rnn - 1):
                layer = self._create_dense_layer(layer, 128, tf.nn.relu, dropout_rate, self.inputs.is_training)
            # shape: batch_size, time_steps, n_rnn_hiddens
            rnn_inputs = tf.layers.dense(layer, n_rnn_hiddens, activation=tf.nn.leaky_relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

            # -------------------------
            # Recurrent layer
            rnn_cells = [tf.nn.rnn_cell.LSTMCell(n_rnn_hiddens, forget_bias=1.0, activation=tf.nn.leaky_relu,
                                                 state_is_tuple=True)
                         for _ in range(n_rnn_layers)]
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells, state_is_tuple=True)

            unstacked_states = tf.unstack(self.inputs.rnn_init_state)  # for each layer
            initial_state_tuple = tuple(tf.nn.rnn_cell.LSTMStateTuple(*tf.unstack(state)) for state in unstacked_states)

            rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.rnn_cell, rnn_inputs,
                                                            initial_state=initial_state_tuple)

            # -------------------------
            # Compress rnn output through multiple dense layer to get the final result
            layer = rnn_outputs
            for i in range(n_dense_after_rnn - 1):
                layer = self._create_dense_layer(layer, 64, tf.nn.leaky_relu, dropout_rate, self.inputs.is_training)
            dense = tf.layers.dense(layer, n_outputs, activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            self.outputs = tf.multiply(dense, self.inputs.game_state.valid_mask)

            # -------------------------
            # Back propagation layer
            one_hot_loss_mask = tf.one_hot(self.inputs.loss_mask, n_outputs)

            self.loss = tf.losses.mean_squared_error(self.inputs.targets,
                                                     self.outputs,
                                                     weights=one_hot_loss_mask)

            # unmasked_loss = tf.nn.softmax_cross_entropy_with_logits_v2(targets=self.targets, logits=outputs)
            # self.loss = tf.reduce_mean(tf.boolean_mask(unmasked_loss, self.loss_mask))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

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

    def predict(self, games, rnn_init_state):
        """
        Give the predicted Q values of a batch of game
        """
        input_dict = {tensor: value for tensor, value in zip(self.inputs.game_state, games)}
        input_dict[self.inputs.rnn_init_state] = rnn_init_state
        input_dict[self.inputs.is_training] = False
        q_values, rnn_state = self.sess.run([self.outputs, self.rnn_state], feed_dict=input_dict)

        return q_values, rnn_state

    def train(self, games, rnn_init_state, targets, loss_mask):
        """
        Train the network with the given game and target values
        games: list of game, each game contains t successive game states
        targets: list of labels, each label contains t successive labels for a game
        """
        input_dict = {tensor: value for tensor, value in zip(self.inputs.game_state, games)}
        input_dict[self.inputs.rnn_init_state] = rnn_init_state
        input_dict[self.inputs.loss_mask] = loss_mask
        input_dict[self.inputs.targets] = targets
        input_dict[self.inputs.is_training] = True
        loss, _ = self.sess.run([self.loss, self.train_step], feed_dict=input_dict)
        return loss

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
