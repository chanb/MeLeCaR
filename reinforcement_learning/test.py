import gym
import envs.cache_bandit

env = gym.make("Cache-Bandit-C30-casa-v0")
print(env)

state = env.reset()
print(state)

done = False
while not done:
  state, reward, done, info = env.step(29)
  print(state)
  print(reward)
  print(done)
  print(info)