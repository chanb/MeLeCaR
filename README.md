# MeLeCaR

We attempt to create a cache eviction policy using multi-label multi-class supervised learning and reinforcement learning. For reinforcement learning context, we treat the problem as a contextual bandit problem, with action space equals to the cache size.  

## Supervised Learning Approach
- GRU Model with sequence length of 5

## Reinforcement Learning Approach
- REINFORCE: Achieve near optimal (We conduct more experiment on this)  
- Actor Critic: Fails to achieve near optimal  

Example training: `python rl_train.py --gamma=0.99 --batch_size=100000 --num_epochs=1500 --algo=reinforce --output_dir=rl_trained/reinforce_abs_norm_EPOCH1500_C100_T100000_ent0.01_home6 --save_interval=10 --output_prefix=gru_reinforce_C100_E1500_T100000_ent0.01_home6 --max_requests=100000 --task_name=home --num_actions=100 --file_index=6 --num_workers=0 --entropy_coef=0.01 |& tee reinforce_EPOCH1500_C100_T100000_ent0.01_home6.txt`

Example evaluation: `python rl_test.py --task_name=mail --file_index=3 --num_actions=30 --max_requests=25000000 --starting_request=0 --input_model=rl_saved_agents/gru_reinforce_C30_E1000_T10000.pkl  --output_name=eval_cheetah3_0-25000000_C30.pkl --output_dir=experiment_results/`

Example hitrate plot: `python plot.py --type=hitrate --experiment_name=casa4_1-1000000_C100 --policy_name="GRU (Ours)" --policy_dir=experiment_results/eval_casa4_1-1000000_C100/ --policy_suffix=eval_casa4_1-1000000_C100.pkl --baseline_result=baseline_results/cache_100/baseline_casa4-1-1000000.pkl --workload_length=1000000`

Example learning curve plot: `python plot.py --type=learning --policy_name=casa6_1-100000_C100 --policy_dir=rl_trained/reinforce_abs_norm_EPOCH1500_C100_T100000_ent0.01_home6/ --policy_suffix=reinforce_EPOCH1500_C100_T100000_ent0.01_home6.txt`