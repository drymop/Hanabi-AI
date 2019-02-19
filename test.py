from game import Game

load_model = ""
n_players = 3

def play(model):
  g = Game(n_players)
  while not g.is_over:
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