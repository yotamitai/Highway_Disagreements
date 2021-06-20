# import gym
# import highway_env
# from matplotlib import pyplot as plt
#
# env = gym.make('highway-v0')
# env.reset()
# for _ in range(3):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, info = env.step(action)
#     env.render()
#
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()


import gym
import highway_env
import numpy as np

from stable_baselines import HER, SAC

env = gym.make("highway-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# SAC hyperparams:
model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))

model.learn(int(2e5))
model.save('her_sac_highway')

# Load saved model
model = HER.load('her_sac_highway', env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(100):
  action, _ = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  episode_reward += reward
  if done or info.get('is_success', False):
    print("Reward:", episode_reward, "Success?", info.get('is_success', False))
    episode_reward = 0.0
    obs = env.reset()