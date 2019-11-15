python -m dl.main logs_rnn ./ppo.gin -b "PPO.policy_fn=@a3c_rnn_fn" \
"PPO.batch_size=4" "PPO.rollout_length=128" "make_atari_env.frame_stack=1"
