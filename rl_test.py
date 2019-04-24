import argparse
import gym
import numpy as np
import torch
import os
import gc
import pickle

from config import *
import rl_envs
from utils.parser_util import str2bool


def test(num_tests, task_name, file_index, num_actions, starting_request, max_requests, input_model, output_dir, output_name):
  assert task_name in TASKS, "Invalid task. Choices: {}".format(TASKS)
  assert file_index in FILE_INDEX, "Invalid file index. Choices: {}".format(FILE_INDEX)
  assert num_actions in CACHE_SIZE, "Invalid number of actions. Choices: {}".format(CACHE_SIZE)
  assert max_requests in MAX_REQUESTS, "Invalid maximum requests allowed. Choices: {}".format(MAX_REQUESTS)
  assert os.path.isfile(input_model), "Input model {} doesn't exist".format(input_model)

  if not os.path.isdir(output_dir):
    print("Constructing directories {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

  num_feature = num_actions * 3

  # Setup environment

  task_name = "Cache-Bandit-C{}-Max{}-{}-{}-v0".format(num_actions, max_requests, task_name, file_index)

  env = gym.make(task_name)

  # Create the model
  model = torch.load(input_model, map_location=MAP_LOCATION)
  model.eval()

  print("Test task: {}".format(task_name))
  print("Input model: {}".format(input_model))
  print("Starting request: {}".format(starting_request))

  hit_rates = []
  for i in range(num_tests):
    print("Performing test {} ==========================================".format(i))
    state = env.reset(starting_request)
    done = False
    hidden_state = model.init_hidden_state(1).to(DEVICE)
    prev_timestep = 0
    info = None

    while not done:
      state = torch.from_numpy(state.reshape(1, 1, num_feature)).float().to(DEVICE)

      with torch.no_grad():
        dist, _, hidden_state = model(state, hidden_state)

      action = dist.sample().cpu().numpy()[0]
      state, _, done, info = env.step(action)
      
      if env._counter - prev_timestep >= 500000:
        print("Current timestep: {}\tHit: {}\tHitrate: {}".format(info["timestep"], info["hit"], info["hit"]/(info["timestep"] - starting_request)))
        prev_timestep = env._counter
        gc.collect()


    final_hitrate = info["hit"]/(info["timestep"] - info["starting_request"])
    with open("{}/{:02d}_{}".format(output_dir.rstrip("/"), i, output_name), 'wb+') as f:
      pickle.dump([env.hitrates, final_hitrate], f)


    print("All requests are processed - Number of hits: {}\tNumber of requests: {}\tHit Ratio: {}".format(info["hit"], info["timestep"] - info["starting_request"], final_hitrate))

    hit_rates.append(final_hitrate)

  hit_rates = np.array(hit_rates)
  print("Average: {}\tStandard Deviation: {}".format(np.average(hit_rates), np.std(hit_rates)))
  with open("{}/final_{}".format(output_dir.rstrip("/"), output_name), 'wb+') as f:
    pickle.dump([hit_rates, np.average(hit_rates), np.std(hit_rates)], f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_tests', type=int, default=10, help='number of tests to perform')

  parser.add_argument("--task_name", type=str, help="the task to learn", default="home", choices=TASKS)
  parser.add_argument("--file_index", type=int, help="the blocktrace file index", default=6, choices=FILE_INDEX)
  parser.add_argument("--num_actions", type=int, help="the number of actions in the task", default=30, choices=CACHE_SIZE)
  parser.add_argument("--max_requests", type=int, help="the maximum number of requests from workload", default=50000, choices=MAX_REQUESTS)
  parser.add_argument("--starting_request", type=int, help="the starting request from workload", default=10000)

  parser.add_argument("--input_model", type=str, help="the full path of model to load", required=True)
  parser.add_argument("--output_dir", type=str, help="the directory to save the results", required=True)
  parser.add_argument("--output_name", type=str, help="the file name to store the results", required=True)

  args = parser.parse_args()

  test(args.num_tests, args.task_name, args.file_index, args.num_actions, args.starting_request, args.max_requests, args.input_model, args.output_dir, args.output_name)
