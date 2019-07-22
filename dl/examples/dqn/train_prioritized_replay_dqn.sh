python -m dl.train logs ./dqn.gin -b "train.Trainer=@PrioritizedReplayDQN" "optim.RMSprop.lr=0.0000625"
