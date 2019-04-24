# MeLeCaR

We attempt to create a cache eviction policy using multi-label multi-class supervised learning and reinforcement learning. For reinforcement learning context, we treat the problem as a multi-armed bandit problem, with action space equals to the cache size.  

## Supervised Learning Approach
- GRU Model with sequence length of 5

## Reinforcement Learning Approach
- REINFORCE: Achieve near optimal  
- Actor Critic: Fails to achieve near optimal  


### Models Training:  
NOTE: Current training are all done on CASA-6  
**Cache Size: 10**  
- Number of trajectories: 1 (length of 10000)
- Number of epochs: 5000
- Workload: casa 6 (First 10000 requests)

**Cache Size: 100**  
- Number of trajectories: 1 (length of 10000)
- Number of epochs: 5000
- Workload: casa 6 (First 50000 requests)


### Models Trained:
**Cache Size: 10**  
- Number of trajectories: 1 (length of 10000)
- Number of epochs: 1000
- Workload: casa 6 (First 10000 requests)

**Cache Size: 30**  
- Number of trajectories: 1 (length of 10000)
- Number of epochs: 1000
- Workload: casa 6 (First 10000 requests)

**Cache Size: 100**  
  - Number of trajectories: 1 (length of 10000)
  - Number of epochs: 1000
  - Workload: casa 6 (First 10000 requests)

  - Number of trajectories: 1 (length of 10000)
  - Number of epochs: 5000
  - Workload: casa 6 (First 10000 requests)


Example evaluation: `python rl_test.py --task_name=mail --file_index=3 --num_actions=30 --max_requests=25000000 --starting_request=0 --input_model=rl_saved_agents/gru_reinforce_C30_E1000_T10000.pkl  --output_name=eval_cheetah3_0-25000000_C30.pkl --output_dir=experiment_results/`


### Evaluation:
#### casa-110108-112108.6.blkparse
**size: 30**  
10000-100000: Done  
OPT: 0.2497888888888889  
LFU: 0.008077777777777777  
LRU: 0.002688888888888889  

10000-500000: Done  
OPT: 0.21428775510204082  
LFU: 0.0077428571428571425  
LRU: 0.0028775510204081633  

10000-1000000: Done  
OPT: 0.19223636363636365  
LFU: 0.00833939393939394  
LRU: 0.0027818181818181817  

whole trace:  
OPT: 0.1975851490600173  
LFU: 0.011217651223157397  
LRU: 0.002791630614331786  

**size: 100**  
10000-100000:  
OPT: 0.5999222222222222  
LFU: 0.5304111111111112  
LRU: 0.29602222222222224  

10000-500000:  
OPT: 0.5138918367346939  
LFU: 0.45006734693877554  
LRU: 0.2333734693877551  

50000-250000000 (1221300)
OPT: 0.46881847211987226  
LFU: 0.4112879718332924  
LRU: 0.19573323507737656  

#### casa-110108-112108.5.blkparse
**size: 30**  
0-100000: Done  
OPT: 0.08841  
LFU: 0.00426  
LRU: 0.00129  

0-500000: Done  
OPT: 0.184366  
LFU: 0.010334  
LRU: 0.002292  

#### casa-110108-112108.4.blkparse
**size: 30**  
0-100000  
OPT: 0.10213  
LFU: 0.00574  
LRU: 0.00181  

0-500000: Done  
OPT: 0.186554  
LFU: 0.007726  
LRU: 0.002288  

#### cheetah.cs.fiu.edu-110108-113008.3.blkparse
**size: 30**  
0-2500000  
OPT: 0.0578644  
LFU: 0.054392  
LRU: 0.0556824  

0-25000000 (max number: 23374516)  
OPT: 0.06550753820956122  
LFU: 0.05630794665438206  
LRU: 0.05876155895591592

**size: 100**  
0-25000000 (max number: 23374516)  
OPT: 0.06807268223222247  
LFU: 0.060437828958683035  
LRU: 0.061514300445835966  
