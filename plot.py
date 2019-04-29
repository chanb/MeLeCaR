import argparse
import os
import pickle
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

from config import *

def plot_hitrate(experiment_name, policy_name, policy_dir, policy_suffix, baseline_result, workload_length, starting_request):
  assert os.path.isdir(policy_dir), "The policy directory {} doesn't exist.".format(policy_dir)
  assert not baseline_result or os.path.isfile(baseline_result), "The result file {} doesn't exist.".format(baseline_result)
  assert workload_length > 0, "The workload length must be greater than 0."
  assert workload_length > starting_request > 0, "The starting request must be greater than 0 and less than workload length."
  
  if not os.path.isdir('plot_hitrate'):
    os.makedirs('plot_hitrate', exist_ok=True)

  # Read all results
  sorted_dir = sorted(os.listdir(policy_dir))
  valid_results = [f for f in sorted_dir if f.endswith(policy_suffix) and not f.startswith("final")]
  assert len(valid_results) > 0, "There is no result file in directory {} that matches the suffix {}".format(policy_dir, policy_suffix)

  policies_checkpoint_hitrates = []
  policies_final_hitrates = []

  for filename in valid_results:
    with open(policy_dir.rstrip("/") + "/" + filename, 'rb') as f:
      policy_checkpoint_hitrates, policy_final_hitrate = pickle.load(f)
      assert workload_length // (SAVE_INTERVAL * len(policy_checkpoint_hitrates)) == 1, "The workload length is shorter than provided policy result."

      policies_checkpoint_hitrates.append(policy_checkpoint_hitrates)
      policies_final_hitrates.append([policy_final_hitrate])

  policies_checkpoint_hitrates = np.array(policies_checkpoint_hitrates)
  policies_final_hitrates = np.array(policies_final_hitrates)

  print("Finished loading policy results")

  with open(baseline_result, 'rb') as f:
    lru_checkpoint_hitrates, lru_final_hitrate, lfu_checkpoint_hitrates, lfu_final_hitrate, opt_checkpoint_hitrates, opt_final_hitrate = pickle.load(f)

  assert workload_length // (SAVE_INTERVAL * len(lru_checkpoint_hitrates)) == 1, "The workload length is shorter than provided baseline results."

  print("Finished loading baseline results")

  x_range = list(range(starting_request - (starting_request % SAVE_INTERVAL) + SAVE_INTERVAL, workload_length + 1, SAVE_INTERVAL))

  # Add one more data point if it's not divisible by save interval
  if workload_length != SAVE_INTERVAL * len(lru_checkpoint_hitrates):
    lru_checkpoint_hitrates.append(lru_final_hitrate)
    lfu_checkpoint_hitrates.append(lfu_final_hitrate)
    opt_checkpoint_hitrates.append(opt_final_hitrate)
    policies_checkpoint_hitrates = np.append(policies_checkpoint_hitrates, policies_final_hitrates, axis=1)
    x_range.append(workload_length)

  # Plot the average and std of policy results
  hitrate_avg = np.average(policies_checkpoint_hitrates, axis=0)
  hitrate_std = np.std(policies_checkpoint_hitrates, axis=0)

  print("Begin plotting")
  plt.plot(x_range, hitrate_avg, label=policy_name, color="black")
  plt.fill_between(x_range, hitrate_avg - hitrate_std, hitrate_avg + hitrate_std, color = 'blue', alpha=0.3, lw=0.001)

  # Plot baseline
  plt.plot(x_range, lru_checkpoint_hitrates, label="LRU", linestyle=":")
  plt.plot(x_range, lfu_checkpoint_hitrates, label="LFU", linestyle="--")
  plt.plot(x_range, opt_checkpoint_hitrates, label="OPT", linestyle="-.")

  plt.xlabel("Requests")
  plt.ylabel("Hit rate")
  plt.title("Experiment: {}".format(experiment_name))
  plt.legend()
  plt.savefig('./plot_hitrate/{0}.png'.format(experiment_name))

  plt.show()


def parse_learning_curve(file):
  if not os.path.isdir('plot_learning_curve'):
    os.makedirs('plot_learning_curve', exist_ok=True)

  prefix = "Number of hits: "
  len_prefix = len(prefix)
  suffix = "Number of requests:"
  rewards = []

  for line in file:
    if prefix not in line:
      continue
    rewards.append(int(line[line.rindex(prefix) + len_prefix:line.index(suffix)].replace(" ", "")))

  return rewards


def plot_learning_curve(policy_name, policy_dir, policy_file):
  full_path = policy_dir.rstrip("/") + "/" + policy_file
  assert os.path.isfile(full_path), "The result file {} doesn't exist.".format(full_path)

  with open(full_path, 'r') as f:
    rewards = parse_learning_curve(f)

  x_range = list(range(len(rewards)))

  plt.plot(x_range, rewards)
  plt.xlabel("Training Iteration")
  plt.ylabel("Total Return")
  plt.title("Learning Curve: {}".format(policy_name))
  plt.savefig('./plot_learning_curve/{0}.png'.format(policy_name))
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--type", help="the plot type", type=str, required=True, choices=PLOTS)
  parser.add_argument("--experiment_name", help="the experiment name", type=str, required=False)
  parser.add_argument("--policy_name", help="the name of the policy", type=str, required=True)
  parser.add_argument("--policy_dir", help="the directory that stores the result file from using the policy", type=str, required=True)
  parser.add_argument("--policy_suffix", help="the result file name suffix from using the policy", type=str, required=True)
  parser.add_argument("--baseline_result", help="the result file from the baselines", type=str, required=False)
  parser.add_argument("--workload_length", help="the length of the workload evaluated", type=int)
  parser.add_argument("--starting_request", help="the starting request of the workload evaluated", type=int, default=1)

  args = parser.parse_args()

  if args.type == PLOT_HITRATE:
    plot_hitrate(args.experiment_name, args.policy_name, args.policy_dir, args.policy_suffix, args.baseline_result, args.workload_length, args.starting_request)
  elif args.type == PLOT_LEARNING:
    plot_learning_curve(args.policy_name, args.policy_dir, args.policy_suffix)