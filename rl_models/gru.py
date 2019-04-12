import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import Categorical
from rl_models.model_init import weight_init
from config import DEVICE


class GRUActorCritic(nn.Module):
  def __init__(self, output_size, input_size, hidden_size=256):
    super(GRUActorCritic, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.relu1 = nn.ReLU()
    self.policy = nn.Linear(hidden_size, output_size)
    self.value = nn.Linear(hidden_size, 1)
    # self.relu2 = nn.ReLU()
    self.apply(weight_init)

  def forward(self, x, h):
    x, h = self.gru(x, h)
    x = self.relu1(x)
    val = self.value(x)
    # val = self.relu2(val)
    dist = self.policy(x).squeeze(0)
    # print(F.softmax(mu, dim=1))
    return Categorical(logits=dist), val, h

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([1, max(1, batchsize), self.hidden_size])


class GRUPolicy(nn.Module):
  def __init__(self, output_size, input_size, hidden_size=256):
    super(GRUPolicy, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.relu1 = nn.ReLU()
    self.policy = nn.Linear(hidden_size, output_size)
    self.apply(weight_init)

  def forward(self, x, h):
    x, h = self.gru(x, h)
    x = self.relu1(x)
    dist = self.policy(x).squeeze(0)
    return Categorical(logits=dist), self.val, h

  def init_hidden_state(self, batch_size=1):
    self.batch_size = batch_size
    self.val = torch.zeros(self.batch_size, 1).to(DEVICE)
    return torch.zeros([1, max(1, batch_size), self.hidden_size])
