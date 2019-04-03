class Reinforce:
  def __init__(self, model, optimizer, baseline=None):
    self.model = model
    self.optimizer = optimizer
    self.baseline = baseline

  def update(self, sampler):
    pass