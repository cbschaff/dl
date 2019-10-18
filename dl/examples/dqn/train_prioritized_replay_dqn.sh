python -m dl.train logs ./dqn.gin -b "train.Trainer=@PrioritizedReplayDQN" \
"optim.RMSprop.lr=0.0000625" "DQN.eval_period=50000"
