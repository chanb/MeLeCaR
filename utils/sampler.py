import torch
import numpy as np
import gym
import multiprocessing as mp
from rl_envs.multiprocessing_env import SubprocVecEnv
from config import *

EPS = 1e-8


# Returns a callable function for SubprocVecEnv
def make_env(env_name):
  def _make_env():
    return gym.make(env_name)
  return _make_env


# This samples from the current environment using the provided model
class Sampler():
  def __init__(self, model, env_name, num_actions, deterministic=False, gamma=0.99, tau=0.3, num_workers=mp.cpu_count() - 1, evaluate=False):
    self.model = model
    self.env_name = env_name
    self.num_actions = num_actions
    self.gamma = gamma
    self.tau = tau
    self.last_hidden_state = None
    self.get_next_action = self.exploit if deterministic else self.random_sample
    self.save_evaluate = self.store_clean if evaluate else lambda action, state, reward: None
    self.reset_evaluate = self.reset_clean if evaluate else lambda: None

    self.reset_storage()

    # This is for multi-processing
    self.num_workers = num_workers
    self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)]) if num_workers > 0 else gym.make(env_name)
    self.max_length = self.envs.get_max_request()

  # Computes the advantage where lambda = tau
  def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
      delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
      gae = delta + gamma * tau * masks[step] * gae
      returns.insert(0, gae + values[step])

    return returns


  # Reset environments
  def reset_envs(self):
    self.envs.reset()


  # Set the current task
  def set_task(self, task):
    tasks = [task for _ in range(self.num_workers)]
    reset = self.envs.reset_task(tasks)


  # Reset the storage
  def reset_storage(self):
    self.actions = []
    self.values = []
    self.states = []
    self.rewards = []
    self.log_probs = []
    self.masks = []
    self.returns = []
    self.advantages = []
    self.hidden_states = []
    self.entropies = []
    self.reset_evaluate()


  # Concatenate storage for more accessibility
  def concat_storage(self):
    # Store in better format
    self.returns = torch.cat(self.returns).to(DEVICE)
    self.values = torch.cat(self.values).to(DEVICE)
    self.log_probs = torch.cat(self.log_probs).to(DEVICE)
    self.states = torch.cat(self.states)
    self.actions = torch.cat(self.actions)
    self.entropies = torch.cat(self.entropies).to(DEVICE)
    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + EPS).to(DEVICE)


  # Concatenate hidden states
  def get_hidden_states(self):
    return torch.stack(self.hidden_states)


  # Insert a sample into the storage
  def insert_storage(self, log_prob, state, action, reward, done, value, hidden_state, entropy):
    self.log_probs.append(log_prob.unsqueeze(0))
    self.states.append(state.unsqueeze(0))
    self.actions.append(action.unsqueeze(0).unsqueeze(0))
    self.rewards.append(torch.Tensor(reward).unsqueeze(1).to(DEVICE))
    self.masks.append(torch.Tensor(1 - done).unsqueeze(1).to(DEVICE))
    self.values.append(value)
    self.entropies.append(entropy.unsqueeze(0))
    self.hidden_states.append(hidden_state)


  # Resets the trajectory to the beginning
  def reset_traj(self, starting_point=0):
    states = self.envs.reset(starting_point)
    return torch.from_numpy(states), torch.zeros([self.num_workers, ]), torch.from_numpy(np.full((self.num_workers, ), -1)), torch.zeros([self.num_workers, ])


  # Generate the state vector for RNN in RL2
  # def generate_state_vector(self, done, reward, num_actions, action, state):
  #   done_entry = done.float().unsqueeze(1)
  #   reward_entry = reward.float().unsqueeze(1)
  #   action_vector = torch.zeros([self.num_workers, num_actions])

  #   # Try to speed up while having some check
  #   if all(action > -1):
  #     action_vector.scatter_(1, action.cpu().unsqueeze(1), 1)
  #   elif any(action > -1):
  #     assert False, 'All processes should be at the same step'
    
  #   state = torch.cat((state, action_vector, reward_entry, done_entry), 1)
  #   state = state.unsqueeze(0)
  #   return state.to(DEVICE)


  def generate_state_vector(self, done, reward, num_actions, action, state):    
    state = state.reshape(1, 1, num_actions * 3).float()
    return state.to(DEVICE)


  # Exploit the best action
  def exploit(self, dist):
    if (len(dist.probs.unique()) == 1):
      action = np.random.randint(0, self.num_actions, size=self.num_workers)
      return torch.from_numpy(action)
    return dist.probs.argmax(dim=-1, keepdim=False)


  # Randomly sample action based on the categorical distribution
  def random_sample(self, dist):
    return dist.sample()


  #TODO: Add code to handle non recurrent case
  # Sample batchsize amount of moves
  def sample(self, batchsize, last_hidden_state=None, stop_at_done=False, starting_point=0):
    state, reward, action, done = self.reset_traj(starting_point)

    hidden_state = last_hidden_state
    if last_hidden_state is None:
      hidden_state = self.model.init_hidden_state(self.num_workers).to(DEVICE)

    # We sample batchsize amount of data
    for i in range(batchsize):
      # Set the vector state
      state = self.generate_state_vector(done, reward, self.num_actions, action, state)

      # Get information from model and take action
      # with torch.no_grad():
      dist, value, next_hidden_state = self.model(state, hidden_state)
      
      # Decide if we should exploit all the time
      action = self.get_next_action(dist)
      if self.num_workers == 0:
        action = action[0]
      next_state, reward, done, info = self.envs.step(action.cpu().numpy())

      log_prob = dist.log_prob(action)
      entropy = dist.entropy().mean().clone()

      if self.num_workers > 0:
        done = done.astype(int)
        reward = torch.from_numpy(reward).float()
        done = torch.from_numpy(done).float()
      else:
        done = torch.tensor([int(done)]).float()
        reward = torch.tensor([reward]).float()

        
      # Store the information
      self.insert_storage(log_prob, state, action, reward, done, value, hidden_state, entropy)
      self.save_evaluate(action, state, reward)

      # Update to the next value
      state = next_state
      state = torch.from_numpy(state).float()
      
      hidden_state = next_hidden_state.to(DEVICE)

      # Grab hidden state for the extra information
      if all(done):
        print(done)
        if self.num_workers > 0:
          info = info[0]
        print("All requests are processed - Number of hits: {}\tNumber of requests: {}\tHit Ratio: {}".format(info["hit"], info["timestep"] - info["starting_request"], info["hit"]/(info["timestep"] - info["starting_request"])))
        if stop_at_done:
          state = self.generate_state_vector(done, reward, self.num_actions, action, state)
          with torch.no_grad():
            _, next_val, _, = self.model(state, hidden_state)

          self.returns = self.compute_gae(next_val.detach(), self.rewards, self.masks, self.values, self.gamma, self.tau)
          return
        state, reward, action, done = self.reset_traj(starting_point)
        hidden_state = self.model.init_hidden_state()
      elif any(done):
        # This is due to environment setting
        # TODO: Allow different trajectory lengths
        assert False, 'All processes be done at the same time'

    ########################################################################
    # self.print_debug()
    ########################################################################

    self.last_hidden_state = hidden_state
    
    if self.num_workers > 0:
      info = info[0]
      
    print("Leftover requests - Number of hits: {}\tNumber of requests: {}\tHit Ratio: {}".format(info["hit"], info["timestep"] - info["starting_request"], info["hit"]/(info["timestep"] - info["starting_request"])))
    # Compute the return
    state = self.generate_state_vector(done, reward, self.num_actions, action, state)
    with torch.no_grad():
      _, next_val, _, = self.model(state, hidden_state)

    self.returns = self.compute_gae(next_val.detach(), self.rewards, self.masks, self.values, self.gamma, self.tau)


  # Storing this for evaluation
  def store_clean(self, action, state, reward):
    self.clean_actions.append(action)
    self.clean_states.append(state)
    self.clean_rewards.append(reward)


  # Reset evaluate information
  def reset_clean(self):
    self.clean_actions = []
    self.clean_states = []
    self.clean_rewards = []


  # Print debugging information
  def print_clean(self):
    for action, state, reward in zip(self.clean_actions, self.clean_states, self.clean_rewards):
      print('action: {} reward: {} state: {}'.format(action, reward, state))
