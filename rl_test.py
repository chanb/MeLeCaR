import argparse
import gym
import torch
import torch.optim as optim
import os

from config import *
from rl_algos.reinforce import Reinforce
from rl_algos.a2c import AdvantageActorCritic
from rl_models.gru import GRUActorCritic, GRUPolicy
from utils.sampler import Sampler
from utils.parser_util import str2bool

def test(algo, model_type, num_tests, task_name, num_actions, starting_request, max_requests, input_dir, input_model):
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)
  assert algo in ALGOS, "Invalid algorithm. Choices: {}".format(ALGOS)
  assert task_name in TASKS, "Invalid task. Choices: {}".format(TASKS)
  assert os.path.isdir(input_dir), "Input directory {} doesn't exist".format(input_dir)

  model_full_path = input_dir.rstrip("/") + "/" + input_model

  assert os.path.isfile(model_full_path), "Input model {} doesn't exist".format(model_full_path)

  num_feature = num_actions * 3

  # Setup environment
  if task_name == CASA:
    task_name = "Cache-Bandit-C{}-Max{}-casa-v0".format(num_actions, max_requests)

  env = gym.make(task_name)

  # Create the model
  model = torch.load(model_full_path, map_location=MAP_LOCATION)
  model.eval()

  print("Input model: {}".format(model_full_path))
  print("Starting request: {}".format(starting_request))

  for i in range(num_tests):
    print("Performing test {} ==========================================".format(i))
    state = env.reset(starting_request)
    done = False
    hidden_state = model.init_hidden_state(1).to(DEVICE)

    while not done:
      state = torch.from_numpy(state.reshape(1, 1, num_feature)).float()
      dist, _, hidden_state = model(state, hidden_state)
      action = dist.sample().cpu().numpy()
      action = action[0]
      state, reward, done, info = env.step(action)

    print("All requests are processed - Number of hits: {}\tNumber of requests: {}\tHit Ratio: {}".format(info["hit"], info["timestep"] - info["starting_request"], info["hit"]/(info["timestep"] - info["starting_request"])))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--algo", help="the rl algorithm to use", type=str, choices=ALGOS, default="reinforce")
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="gru", help="the model architecture to train")
  parser.add_argument('--num_tests', type=int, default=10, help='number of tests to perform')

  parser.add_argument("--task_name", type=str, help="the task to learn", default="casa", choices=TASKS)
  parser.add_argument("--num_actions", type=int, help="the number of actions in the task", default=30)
  parser.add_argument("--max_requests", type=int, help="the maximum number of requests from workload", default=50000)
  parser.add_argument("--starting_request", type=int, help="the starting request from workload", default=10000)

  parser.add_argument("--input_dir", type=str, help="the directory to load the models from", required=True)
  parser.add_argument("--input_model", type=str, help="the model to load", required=True)

  args = parser.parse_args()

  test(args.algo, args.model_type, args.num_tests, args.task_name, args.num_actions, args.starting_request, args.max_requests, args.input_dir, args.input_model)
