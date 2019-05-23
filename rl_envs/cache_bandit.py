import numpy as np
import pandas as pd
import random
import gc
from sklearn.preprocessing import normalize
from collections import defaultdict
from config import ENV_LOCATION_PREFIX, SAVE_INTERVAL

import gym
from gym import spaces

class CacheBandit(gym.Env):
  def __init__(self, cache_size, workload, max_requests=-1):
    assert cache_size < max_requests, "There is nothing fun about this environment! Cache size ({}) should be smaller than maximum requests ({})".format(cache_size, max_requests)
    print("WORKLOAD: {} CACHE SIZE: {} MAX_REQUESTS: {}".format(workload, cache_size, max_requests))
    self._hit = 0
    self.cache_size = cache_size
    self.workload = ENV_LOCATION_PREFIX + workload
    self.max_requests = max_requests
    self.hitrates = []

    self.action_space = spaces.Discrete(self.cache_size)
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.cache_size, 3), dtype=np.float32)

    self._starting_request = 0
    df = pd.read_csv(self.workload, sep=' ',header = None)
    df.columns = ['timestamp','pid','pname','blockNo', \
                  'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']

    # Normalize abs position
    self._stream = df['blockNo'].values[:max_requests]
    self._stream = self._stream / max(self._stream)

    self._size = len(self._stream)
    self._counter = 0

    self._lfu = defaultdict(int)
    self._lru = []
    self._cache = []

    self._fill_until_evict()

  # TODO: What if the state vector only contains LBA and frequency, but ordered in recency
  def _compute_state(self):
    recency = []
    frequency = []
    recency_dict = defaultdict(int)
    
    # Compute the recency and frequency order of each page in cache
    for time in range(len(self._lru)):
      recency_dict[self._lru[time]] = time + 1
    for block in self._cache:
      recency.append(recency_dict[block])
      frequency.append(self._lfu[block])

    block_num = self._cache[:]
    
    # This shouldn't happen but this pads the 0's to empty slots in cache
    if len(recency) < self.cache_size:
      for _ in range(0, self.cache_size - len(recency)):
        recency.append(0)
        frequency.append(0)
        block_num.append(0)

    block_num = np.expand_dims(np.array(block_num), axis=1)
    
    # Columns: recency, frequency, block number
    # Row: cache location
    raw = np.column_stack((recency, frequency))
    normalized = normalize(raw, axis=0)

    final_state = np.append(normalized, block_num, axis=1)

    # Normalize rel position
    # raw = np.column_stack((recency, frequency, block_num))
    # final_state = normalize(raw, axis=0)

    return final_state


  # TODO: What if we change the reward to be -1 instead of 0 when we have immediate eviction?
  # TODO: What if we normalize the reward?
  # Find the next time where we need to perform an eviction
  def _fill_until_evict(self):
    while self._counter < self._size:
      request_block = self._stream[self._counter]
      self._lfu[request_block] += 1

      if request_block in self._cache:
        # Cache Hit
        self._lru.remove(request_block)
        self._lru.append(request_block)
        self._counter += 1
        self._hit += 1
      else:
        # Cache Miss
        if len(self._cache) == self.cache_size:
          # Agent needs to choose victim block for request block
          return self._counter >= self._size
        else:
          # Cache is not full
          self._cache.append(request_block)
          self._lru.append(request_block)
          self._counter += 1
      if self._counter % SAVE_INTERVAL == 0:
        self.hitrates.append(self._hit / (self._counter - self._starting_request))
    return self._counter >= self._size
    

  # Reset to a specified starting point
  def reset(self, starting_request=0):
    assert starting_request < self._size, "Starting point ({}) is after the request stream ({})".format(starting_point, self._size)
    self._hit = 0
    self._lfu = defaultdict(int)
    self._lru = []
    self._cache = []

    env_done = True
    while env_done:
      print("Reset starting request to {}".format(starting_request))
      self._counter = starting_request
      self._starting_request = starting_request
      self.hitrates = []
      env_done = self._fill_until_evict()
      assert not env_done or starting_request > 0, "This workload {} with max request {} and cache size {} doesn't need to evict any pages.".format(self.workload, self.max_requests, self.cache_size)
      starting_request = random.randint(0, max(0, starting_request - 1))
    return self._compute_state()

  
  def step(self, action):
    assert self.action_space.contains(action)

    # Clear out victim block in cache
    victim_block = self._cache[action]
    request_block = self._stream[self._counter]
    
    self._lru.remove(victim_block)
    del self._lfu[victim_block]
    self._cache.remove(victim_block)

    # Insert the request block
    self._cache.append(request_block)
    self._lru.append(request_block)
    self._lfu[request_block] = 1
    self._counter += 1
    if self._counter % SAVE_INTERVAL == 0:
      self.hitrates.append(self._hit / (self._counter - self._starting_request))
    curr_timestep = self._counter
    
    # Find the next timestep where we need to evict again
    done = self._fill_until_evict()
    state = self._compute_state()
    
    return state, (self._counter - curr_timestep), done, {"workload": self.workload, "timestep": self._counter, "hit": self._hit, "starting_request": self._starting_request}


  def get_max_request(self):
    return self._size
