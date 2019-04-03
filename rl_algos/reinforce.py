import torch

class Reinforce:
  def __init__(self, model, optimizer):
    self.model = model
    self.optimizer = optimizer


  def update(self, sampler):
    states = sampler.states
    actions = sampler.actions
    log_probs = sampler.log_probs
    returns = sampler.returns
    advantages = sampler.advantages
    values = sampler.values
    hidden_states = sampler.get_hidden_states()

    # loss = Adv_t * log_prob(A_t | S_t)
    policy_loss = []
    for log_prob, advantage in zip(log_probs, advantages):
      policy_loss.append(-log_prob * advantage)

    self.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    self.optimizer.step()