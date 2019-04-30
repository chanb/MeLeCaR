# MeLeCaR

We attempt to create a cache eviction policy using multi-label multi-class supervised learning and reinforcement learning. For reinforcement learning context, we treat the problem as a contextual bandit problem, with action space equals to the cache size.  

Please download the FIU workload files from <http://sylab-srv.cs.fiu.edu/doku.php?id=projects:iodedup:start>

Hyperparameters settings are provided in appendix of the PDF.

## Supervised Learning Approach

We formulate this problem as a multi-label binary-class classification. Due to the nature of the problem, we decide to use GRU model with sequence length of 5.

`reptile.py` is a meta-learning algorithm that optimizes for a good initialization for the model. The performance has not been tested yet.

## Reinforcement Learning Approach

We compare OPT, LRU, LFU, and our approach. The results for OPT, LRU, and LFU can be obtained through `generate_baseline.ipynb`. The second last cell loads the blocktrace file and uses logical block numbers the request page. The last cell runs OPT, LRU, and LFU on the specified range of the loaded workload, and store it into a file that can be used for plotting.

### **Environment**

The formulation of the cache replacement problem can be found in the PDF. To get the environment to work, please move the workload blocktrace files `cheetah-3`, `casa-4`, `casa-5`, `casa-6` to `rl_envs` directory.

### **Models**

The models are stored in `rl_models`. We currently use GRU as our default architecture choice. `ActorCritic` model consists of an actor and a critic, which corresponds to the policy and value function. `Policy` only consists of an actor. The models only work for the corresponding algorithm.

### **Algorithms**

We support both A2C and REINFORCE algorithms (See `rl_algos`). We allow using either `SGD` or `Adam` optimizer. Based on quick training, we conclude the following:

- REINFORCE: Achieve near optimal (We conduct more experiment on this)  
- Actor Critic: Fails to achieve near optimal  

### **Scripts**

There are three scripts that are relevant to training and evaluating a RL agent: `rl_train.py`, `rl_test.py`, and `plot.py`. `requirements.txt` consists of libraries used in the code. Simply run `pip install -r requirements.txt` to install the required libraries.

- `rl_train.py`: This script trains a RL agent given a specific workload and cache size.  

  Example training: `python rl_train.py --gamma=0.99 --batch_size=100000 --num_epochs=1500 --algo=reinforce --output_dir=rl_trained/reinforce_abs_norm_EPOCH1500_C100_T100000_ent0.01_home6 --save_interval=10 --output_prefix=gru_reinforce_C100_E1500_T100000_ent0.01_home6 --max_requests=100000 --task_name=home --num_actions=100 --file_index=6 --num_workers=0 --entropy_coef=0.01 |& tee reinforce_EPOCH1500_C100_T100000_ent0.01_home6.txt`

- `rl_test.py`: This script loads a RL agent and test on specificed workload.

  Example evaluation: `python rl_test.py --task_name=mail --file_index=3 --num_actions=30 --max_requests=25000000 --starting_request=0 --input_model=rl_saved_agents/gru_reinforce_C30_E1000_T10000.pkl  --output_name=eval_cheetah3_0-25000000_C30.pkl --output_dir=experiment_results/`

- `plot.py`: This script plots the results of training process and evaluation. You can specify the plot type by providing `hitrate` for evaluation or `learning` for learning curve. For hit rate plots, you will also need to provide the baseline results, which is generated using `generate_baseline.ipynb`.
  
  Example hitrate plot: `python plot.py --type=hitrate --experiment_name=casa4_1-1000000_C100 --policy_name="GRU (Ours)" --policy_dir=experiment_results/eval_casa4_1-1000000_C100/ --policy_suffix=eval_casa4_1-1000000_C100.pkl --baseline_result=baseline_results/cache_100/baseline_casa4-1-1000000.pkl --workload_length=1000000`

  Example learning curve plot: `python plot.py --type=learning --policy_name=casa6_1-100000_C100 --policy_dir=rl_trained/reinforce_abs_norm_EPOCH1500_C100_T100000_ent0.01_home6/ --policy_suffix=reinforce_EPOCH1500_C100_T100000_ent0.01_home6.txt`

The arguments of each script can be found by using `--help`.
