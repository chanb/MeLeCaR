import torch

class AdvantageActorCritic:
  def __init__(self, model, optimizer, critic_coef=0.5, actor_coef=0.5, entropy_coef=0.001):
    self.model = model
    self.optimizer = optimizer
    self.critic_coef = critic_coef
    self.actor_coef = actor_coef
    self.entropy_coef = entropy_coef


  def update(self, sampler):
    log_probs = sampler.log_probs
    advantages = sampler.advantages
    entropies = sampler.entropies

    actor_loss = []
    for log_prob, advantage in zip(log_probs, advantages):
      actor_loss.append(-log_prob * advantage)
    actor_loss = torch.cat(actor_loss).sum()

    # Add value loss
    critic_loss = advantages.pow(2).mean()

    mean_entropy = entropies.mean()
    
    loss = self.critic_coef * critic_loss + self.actor_coef * actor_loss - self.entropy_coef * mean_entropy
    print(loss)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  