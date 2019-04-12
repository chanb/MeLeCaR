import torch

class Reinforce:
  def __init__(self, model, optimizer):
    self.model = model
    self.optimizer = optimizer


  def update(self, sampler):
    log_probs = sampler.log_probs
    returns = sampler.returns

    policy_loss = []
    for log_prob, return_t in zip(log_probs, returns):
      policy_loss.append(-log_prob * return_t)

    policy_loss = torch.cat(policy_loss).sum()
    print(policy_loss)
    
    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()
  
