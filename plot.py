import argparse
import os
import pickle
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

from config import *

def plot_hitrate(experiment_name, policy_name, policy_result, baseline_result, workload_length, starting_request):
  assert os.path.isfile(policy_result), "The result file {} doesn't exist.".format(policy_result)
  assert not baseline_result or os.path.isfile(baseline_result), "The result file {} doesn't exist.".format(baseline_result)
  assert workload_length > 0, "The workload length must be greater than 0."
  assert workload_length > starting_request > 0, "The starting request must be greater than 0 and less than workload length."
  
  with open(policy_result, 'rb') as f:
    policy_checkpoint_hitrates, policy_final_hitrate = pickle.load(f)
  

  with open(baseline_result, 'rb') as f:
    lru_checkpoint_hitrates, lru_final_hitrate, lfu_checkpoint_hitrates, lfu_final_hitrate, opt_checkpoint_hitrates, opt_final_hitrate = pickle.load(f)

  assert workload_length // (SAVE_INTERVAL * len(lru_checkpoint_hitrates)) == 1, "The workload length is shorter than provided baseline results."
  assert workload_length // (SAVE_INTERVAL * len(policy_checkpoint_hitrates)) == 1, "The workload length is shorter than provided policy result."

  x_range = list(range(starting_request + ((starting_request + SAVE_INTERVAL) % SAVE_INTERVAL), workload_length + 1, SAVE_INTERVAL))

  if workload_length != SAVE_INTERVAL * len(lru_checkpoint_hitrates):
    lru_checkpoint_hitrates.append(lru_final_hitrate)
    lfu_checkpoint_hitrates.append(lfu_final_hitrate)
    opt_checkpoint_hitrates.append(opt_final_hitrate)
    policy_checkpoint_hitrates.append(policy_final_hitrate)
    x_range.append(workload_length)


  plt.plot(x_range, lru_checkpoint_hitrates, label="LRU", linestyle=":")
  plt.plot(x_range, lfu_checkpoint_hitrates, label="LFU", linestyle="--")
  plt.plot(x_range, opt_checkpoint_hitrates, label="OPT", linestyle="-.")
  plt.plot(x_range, policy_checkpoint_hitrates, label=policy_name)

  plt.xlabel("Requests")
  plt.ylabel("Hit rate")
  plt.title("Experiment: {}".format(experiment_name))
  plt.legend()

  plt.show()


def plot_learning_curve(policy_name, policy_result):
  assert os.path.isfile(policy_result), "The result file {} doesn't exist.".format(policy_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--type", help="the plot type", type=str, required=True, choices=PLOTS)
  parser.add_argument("--experiment_name", help="the experiment name", type=str, required=False)
  parser.add_argument("--policy_name", help="the name of the policy", type=str, required=True)
  parser.add_argument("--policy_result", help="the result file from using the policy", type=str, required=True)
  parser.add_argument("--baseline_result", help="the result file from the baselines", type=str, required=False)
  parser.add_argument("--workload_length", help="the length of the workload evaluated", type=int)
  parser.add_argument("--starting_request", help="the starting request of the workload evaluated", type=int, default=1)

  args = parser.parse_args()

  if args.type == PLOT_HITRATE:
    plot_hitrate(args.experiment_name, args.policy_name, args.policy_result, args.baseline_result, args.workload_length, args.starting_request)
  elif args.type == PLOT_LEARNING:
    plot_learning_curve(args.policy_name, args.policy_result)