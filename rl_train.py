import argparse
import torch
import torch.optim as optim
import os
import random

from config import *
from rl_algos.reinforce import Reinforce
from rl_algos.a2c import AdvantageActorCritic
from rl_models.gru import GRUActorCritic, GRUPolicy
from utils.sampler import Sampler
from utils.parser_util import str2bool

def train(algo, opt, model_type, batch_size, learning_rate, num_epochs, stop_at_done, gamma, tau, num_workers, task_name, file_index, num_actions, max_requests, starting_request, random_start, critic_coef, actor_coef, entropy_coef, output_dir, output_prefix, save_interval):
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)
  assert opt in OPT_TYPES, "Invalid optimizer type. Choices: {}".format(OPT_TYPES)
  assert algo in ALGOS, "Invalid algorithm. Choices: {}".format(ALGOS)
  assert task_name in TASKS, "Invalid task. Choices: {}".format(TASKS)
  assert file_index in FILE_INDEX, "Invalid file index. Choices: {}".format(FILE_INDEX)
  assert num_actions in CACHE_SIZE, "Invalid number of actions. Choices: {}".format(CACHE_SIZE)
  assert max_requests in MAX_REQUESTS, "Invalid maximum requests allowed. Choices: {}".format(MAX_REQUESTS)
  assert num_workers >= 0, "Invalid number of workers ({}). Must be at least 0.".format(num_workers)
  assert num_epochs >= 1, "Invalid number of epochs ({}). Must be at least 1.".format(num_epochs)
  assert 1 <= save_interval <= num_epochs, "Invalid save interval ({}). Must be between 1 and {}".format(save_interval, num_epochs)

  num_feature = num_actions * 3

  # Setup environment
  task_name = "Cache-Bandit-C{}-Max{}-{}-{}-v0".format(num_actions, max_requests, task_name, file_index)

  opt_construct = optim.Adam if opt == OPT_ADAM else optim.SGD

  # Create the model
  if (model_type == GRU and algo == REINFORCE):
    model = GRUPolicy(num_actions, num_feature)
    # Set the optimizer
    optimizer = opt_construct(model.parameters(), lr=learning_rate)
    agent = Reinforce(model, optimizer, entropy_coef)
  elif(model_type == GRU and algo == A2C):
    model = GRUActorCritic(num_actions, num_feature)
    # Set the optimizer
    optimizer = opt_construct(model.parameters(), lr=learning_rate)
    agent = AdvantageActorCritic(model, optimizer, critic_coef, actor_coef, entropy_coef)

  model = model.to(DEVICE)
  model.train()

  # Setup sampler
  sampler = Sampler(model, task_name, num_actions, deterministic=False, gamma=gamma, tau=tau, num_workers=num_workers)

  def _random_start(max_request):
    return random.randint(0, max(0, max_request - 1 - num_actions))

  get_starting_point = _random_start if random_start else lambda x: starting_request

  print(optimizer)
  print(model)
  print("Stop after singlefull trajectory is completed for each epoch: {}".format(stop_at_done))
  print("Output Directory: {}".format(output_dir))
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

  for epoch in range(num_epochs):
    print("EPOCH {} ==========================================".format(epoch))
    sampler.reset_storage()
    sampler.last_hidden_state = None

    if (MAP_LOCATION == CUDA):
      torch.cuda.empty_cache()

    sampler.sample(batch_size, stop_at_done=stop_at_done, starting_point=get_starting_point(sampler.max_length))
    sampler.concat_storage()
    agent.update(sampler)

    if ((epoch + 1) % save_interval == 0):
      out_file = '{}/{}_{}.pkl'.format(output_dir.rstrip("/"), output_prefix, epoch)
      print("Saving model as {}".format(out_file))
      torch.save(model, out_file)
  
  print("DONE")

  sampler.envs.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--algo", help="the rl algorithm to use", type=str, choices=ALGOS, default="reinforce")
  parser.add_argument("--opt", help="the optimizer to use", type=str, choices=OPT_TYPES, default=OPT_ADAM)
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="gru", help="the model architecture to train")
  parser.add_argument("--batch_size", type=int, help="batch size (number of timesteps taken)", default=128)
  parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-3)
  parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument('--tau', type=float, default=1, help='GAE parameter (default: 1)')
  parser.add_argument('--stop_at_done', type=str2bool, default=True, help='whether or not to sample after one complete trajectory per epoch')

  parser.add_argument("--critic_coef", type=float, default=0.5, help="the contribution of critic loss")
  parser.add_argument("--actor_coef", type=float, default=0.5, help="the contribution of actor loss")
  parser.add_argument("--entropy_coef", type=float, default=0.001, help="the contribution of entropy")

  parser.add_argument("--num_workers", type=int, help="the number of async processes", default=1)
  parser.add_argument("--task_name", type=str, help="the task to learn", default="home", choices=TASKS)
  parser.add_argument("--file_index", type=int, help="the blocktrace file index", default=6, choices=FILE_INDEX)
  parser.add_argument("--num_actions", type=int, help="the number of actions in the task", default=30, choices=CACHE_SIZE)
  parser.add_argument("--max_requests", type=int, help="the maximum number of requests from workload", default=50000, choices=MAX_REQUESTS)
  parser.add_argument("--starting_request", type=int, help="the starting request of the workload", default=0)
  parser.add_argument("--random_start", type=str2bool, default=False, help="whether to use a random index as a starting point")

  parser.add_argument("--output_dir", type=str, help="the directory to save the models", required=True)
  parser.add_argument("--output_prefix", type=str, help="the model prefix to save", required=True)
  parser.add_argument("--save_interval", type=int, help="how often to save the model", default=50)

  args = parser.parse_args()

  train(args.algo, args.opt, args.model_type, args.batch_size, args.learning_rate, args.num_epochs, args.stop_at_done, args.gamma, args.tau, args.num_workers, args.task_name, args.file_index, args.num_actions, args.max_requests, args.starting_request, args.random_start, args.critic_coef, args.actor_coef, args.entropy_coef, args.output_dir, args.output_prefix, args.save_interval)
