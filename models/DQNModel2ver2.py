import numpy as np
import os
import tensorflow as tf


class Model(object):
    """
    DQN model that takes in a sequence of Hanabi game states and output the Q value of each action

    """
    
    def __init__(self, n_hiddens, n_rnn_layers, n_outputs, learn_rate=2e-4, n_players=3, hand_size=5, n_ranks=5, n_suits=5):

        self.graph = tf.Graph()
        self.n_hiddens = n_hiddens
        self.n_rnn_layers = n_rnn_layers
        self.n_outputs = n_outputs

        n_tile_types = n_ranks * n_suits

        with self.graph.as_default():
            #-------------------------
            # Input placeholders

            with tf.variable_scope('inputs'):
                # in_cur_player: current player (relative to this player)
                # Dimensions: (batch_size x time_steps)
                self.in_cur_player = tf.placeholder(tf.int32, shape=[None, None], name='cur_player')

                # The number of undiscarded tiles for each type
                self.in_remain_tiles = tf.placeholder(tf.float32, shape=[None, None, n_tile_types], name='remain_tiles')
                
                # in_hands[player][tileIndex] is the id of the tile (0 to n_tile_types - 1) 
                # player are counted from the player immediately after this player
                # excluding this player's hand
                # Dimensions: (batch_size x time_steps x n_players-1, hand_size)
                self.in_hands = tf.placeholder(tf.int32, shape=[None, None, n_players-1, hand_size], name='hands')
                
                # input_hints [player] [tileIndex] is a one-hot encoded vector 
                # representing the possible types of a tile in player's hand
                # Dimensions: (batch_size x n_players, hand_size, n_tile_types)
                self.in_hints = tf.placeholder(tf.float32, shape=[None, None, n_players, hand_size, n_tile_types], name='hints')

                self.in_hint_tokens = tf.placeholder(tf.float32, shape=[None, None], name='hint_tokens')
                self.in_fuse_tokens = tf.placeholder(tf.float32, shape=[None, None], name='fuse_tokens')

                # firework height of each suit
                # Dimensions: (batch_size, time_step, n_suits)
                self.in_fireworks = tf.placeholder(tf.int32, shape=[None, None, n_suits], name='input_fireworks')
                
                self.in_last_actions = tf.placeholder(tf.int32, shape=[None, None], name='last_action')

                self.initial_state = tf.placeholder(tf.float32, shape=[n_rnn_layers, 2, None, n_hiddens], name='rnn_initial_state') # (n_layers, 2, batch_size, hidden_size)
                self.valid_mask = tf.placeholder(tf.float32, shape=[None, None, n_outputs], name='valid_mask') # valid actions
                self.loss_mask = tf.placeholder(tf.float32, shape=[None,None, n_outputs], name='loss_mask') # outputs to be accounted for in loss calculation
                self.targets = tf.placeholder(tf.float32, shape=[None, None, n_outputs], name='targets')

            #-------------------------
            # Preprocess input heads into 1 large input array:

            time_steps = tf.shape(self.in_cur_player)[1]

            # Create onehot vectors for current player's number
            onehot_cur_player = tf.one_hot(self.in_cur_player, n_players)

            # Create onehot vectors for hands
            onehot_hands = tf.one_hot(self.in_hands, n_tile_types)

            # flatten hands and hints
            hands_hints = tf.concat([onehot_hands, self.in_hints], axis=2)
            hands_hints = tf.reshape(hands_hints, [-1, time_steps, hands_hints.shape[2] * hands_hints.shape[3] * hands_hints.shape[4]])

            # subtract the seen tiles from other players
            remain_tiles = self.in_remain_tiles - tf.reduce_sum(onehot_hands, axis=[2,3])

            # normalize integral quantities into (0, 1) range
            # remain_tiles = tf.scalar_mul(tf.constant(1/MAX_N_PER_TYPES), remain_tiles)
            # hint_tokens  = tf.scalar_mul(tf.constant(1/MAX_HINT_TOKENS), self.in_hint_tokens)
            # fuse_tokens  = tf.scalar_mul(tf.constant(1/MAX_FUSE_TOKENS), self.in_fuse_tokens)
            hint_tokens = self.in_hint_tokens
            fuse_tokens = self.in_fuse_tokens
            onehot_fireworks = tf.one_hot(self.in_fireworks, n_ranks, axis=2) # shape: [batch_size, time_steps, n_ranks, n_suits]
            onehot_fireworks = tf.reshape(onehot_fireworks, [-1, time_steps, n_tile_types])

            # reshape hint_tokens and fuse_tokens from float to array of 1 element
            hint_tokens =  tf.reshape(hint_tokens, [-1, time_steps, 1])
            fuse_tokens =  tf.reshape(fuse_tokens, [-1, time_steps, 1])

            # one hot vector for last action
            one_hot_last_actions = tf.one_hot(self.in_last_actions, n_outputs-1)

            # Processed input concat into a 2D array (batch_size x num_features)
            concated_inputs = tf.concat([onehot_cur_player, remain_tiles, hands_hints, onehot_fireworks, hint_tokens, fuse_tokens, one_hot_last_actions], axis=-1)

            # compress the sparse input through a dense layer
            rnn_inputs = tf.layers.dense(concated_inputs, n_hiddens * 2, activation=tf.nn.sigmoid)
            rnn_inputs = tf.layers.dense(rnn_inputs, n_hiddens, activation=tf.nn.sigmoid)

            #--------------------------
            # Recurrent layer

            rnn_single_cell = tf.nn.rnn_cell.LSTMCell(n_hiddens, forget_bias=1.0, activation=tf.nn.sigmoid, state_is_tuple=True)
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_single_cell] * n_rnn_layers, state_is_tuple=True)

            unstacked_states = tf.unstack(self.initial_state) # for each layer
            initial_state_tuple = tuple(tf.nn.rnn_cell.LSTMStateTuple(*tf.unstack(state)) for state in unstacked_states)

            rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.rnn_cell, rnn_inputs, initial_state=initial_state_tuple)
            dense = tf.layers.dense(rnn_outputs, n_outputs, activation=tf.nn.sigmoid)
            self.outputs = tf.multiply(dense, self.valid_mask)

            #--------------------------
            # Back propagation layer
            self.loss = tf.losses.mean_squared_error(self.targets, self.outputs, weights=self.loss_mask)

            #unmasked_loss = tf.nn.softmax_cross_entropy_with_logits_v2(targets=self.targets, logits=outputs)
            #self.loss = tf.reduce_mean(tf.boolean_mask(unmasked_loss, self.loss_mask))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

        #---------------------------------
        # Initialize graph
        self.sess = tf.Session(graph=self.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))

        #---------------------------------
        # Save and load operations
        self.saver = None

    def predict(self, games, initial_state):
        '''
        Give the predicted Q values of a batch of game
        '''
        input_dict = {
            self.initial_state: initial_state,
            self.in_cur_player: games.cur_player,
            self.in_remain_tiles: games.remain_tiles,
            self.in_hands: games.hands,
            self.in_hints: games.hints,
            self.in_hint_tokens: games.hint_tokens,
            self.in_fuse_tokens: games.fuse_tokens,
            self.in_fireworks: games.fireworks,
            self.in_last_actions: games.last_actions,
            self.valid_mask: games.valid_mask
        }

        # run
        q_values, rnn_state = self.sess.run([self.outputs, self.rnn_state], feed_dict=input_dict)

        return q_values, rnn_state

    def train(self, games, initial_state, targets):
        '''
        Train the network with the given game and target values
        games: list of game, each game contains t successive game states
        targets: list of labels, each label contains t successive labels for a game
        '''
        input_dict = {
            self.initial_state: initial_state,
            self.in_cur_player: games.cur_player,
            self.in_remain_tiles: games.remain_tiles,
            self.in_hands: games.hands,
            self.in_hints: games.hints,
            self.in_hint_tokens: games.hint_tokens,
            self.in_fuse_tokens: games.fuse_tokens,
            self.in_fireworks: games.fireworks,
            self.in_last_actions: games.last_actions,
            self.valid_mask: games.valid_mask,
            self.loss_mask: games.loss_mask,
            self.targets: targets
        }
        loss, _ = self.sess.run([self.loss, self.train_step], feed_dict=input_dict)
        return loss


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.ckpt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        if self.saver == None:
            self.saver = tf.train.Saver(self.graph.get_collection('variables'), max_to_keep=10000000)
        with self.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.ckpt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath + '.meta'):
            raise("No model in path {}".format(filepath))
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)