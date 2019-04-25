from gym.envs.registration import register
from config import *

for max_requests in MAX_REQUESTS:
  for cache_size in CACHE_SIZE:
    for file_index in FILE_INDEX:
      for task in TASKS:
        register(
          "Cache-Bandit-C{}-Max{}-{}-{}-v0".format(cache_size, max_requests, task, file_index),
          entry_point="rl_envs.cache_bandit:CacheBandit",
          kwargs={"cache_size": cache_size, "workload": TASK_FILES[task].format(file_index), "max_requests": max_requests}
        )