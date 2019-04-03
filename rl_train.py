import argparse
import torch
import torch.optim as optim

from config import *
from rl_algos.reinforce import Reinforce
from rl_models.gru import GRUActorCritic
from utils.sampler import Sampler
from utils.parser_util import str2bool

def train(algo, model_type, batch_size, learning_rate, num_epochs, gamma, tau, task_name, num_actions):
  assert model_type in MODEL_TYPES, "Invalid model type. Choices: {}".format(MODEL_TYPES)
  assert task_name in TASKS, "Invalid task. Choices: {}".format(TASKS)

  num_feature = num_actions * 3

  # Setup environment
  if task_name == CASA:
    task_name = "Cache-Bandit-C{}-casa-v0".format(num_actions)

  # Create the model
  if (model_type == GRU):
    model = GRUActorCritic(num_actions, num_feature)

  model = model.to(DEVICE)

  # Set the optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Setup sampler
  sampler = Sampler(model, task_name, num_actions, deterministic=False, gamma=gamma, tau=tau, num_workers=1)
  agent = Reinforce(model, optimizer)
  
  for epoch in range(num_epochs):
    print("EPOCH {} ==========================================".format(epoch))
    sampler.reset_storage()
    sampler.last_hidden_state = None

    sampler.sample(batch_size)
    agent.update(sampler)

  sampler.envs.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--algo", help="the rl algorithm to use", type=str, choices=ALGOS, default="reinforce")
  parser.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="gru", help="the model architecture to train")
  parser.add_argument("--batch_size", type=int, help="batch size", default=128)
  parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-3)
  parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
  parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)
  parser.add_argument('--tau', type=float, default=1, help='GAE parameter (default: 1)')
  parser.add_argument('--baseline', type=float, default=False, help='whether or not to use baseline')


  parser.add_argument("--task_name", type=str, help="the task to learn", default="casa", choices=TASKS)
  parser.add_argument("--num_actions", type=int, help="the number of actions in the task", default=30)

  args = parser.parse_args()

  train(args.algo, args.model_type, args.batch_size, args.learning_rate, args.num_epochs, args.gamma, args.tau, args.task_name, args.num_actions)
