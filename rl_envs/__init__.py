from gym.envs.registration import register

for max_requests in [10000, 50000, 100000, 500000]:
  for cache_size in [10, 30]:
    register(
      "Cache-Bandit-C{}-Max{}-{}-v0".format(cache_size, max_requests, "casa"),
      entry_point="rl_envs.cache_bandit:CacheBandit",
      kwargs={"cache_size": cache_size, "workload": "casa-110108-112108.6.blkparse", "max_requests": max_requests}
    )