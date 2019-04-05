from gym.envs.registration import register

for cache_size in [10, 30]:
  register(
    "Cache-Bandit-C{}-{}-v0".format(cache_size, "casa"),
    entry_point="rl_envs.cache_bandit:CacheBandit",
    kwargs={"cache_size": cache_size, "workload": "casa-110108-112108.6.blkparse", "max_requests": 10000}
  )