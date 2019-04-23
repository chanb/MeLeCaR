import torch

class Reinforce:
  def __init__(self, model, optimizer, entropy_coef=0):
    self.model = model
    self.optimizer = optimizer
    self.entropy_coef = entropy_coef


  def update(self, sampler):
    #TODO: Add entropy term
    log_probs = sampler.log_probs
    returns = sampler.returns
    ent = sampler.entropies

    policy_loss = []
    for log_prob, return_t in zip(log_probs, returns):
      policy_loss.append(-log_prob * return_t)

    policy_loss = torch.cat(policy_loss).sum() - self.entropy_coef * ent.mean()
    print(policy_loss)
    
    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()
  
