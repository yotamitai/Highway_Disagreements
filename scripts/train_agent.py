import gym
import highway_env
from highway_disagreements.envs.highway_env_local import LocalHighwayEnv
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory
from scripts.experiments import evaluate
from stable_baselines import DQN


# def configure_env():
#     env = gym.make('highway-v0')
#     env.configure({
#         "lanes_count": 4,
#         "vehicles_count": 40,
#         "observation": {
#             "type": "GrayscaleObservation",
#             "observation_shape": (128, 64),
#             "stack_size": 4,
#             "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
#             "scaling": 1.75,
#         },
#         "policy_frequency": 2,
#         "duration": 40,
#     })
#     env.reset()
#     return env
#
# env = configure_env()


env = gym.make('highway_local-v0')
agent = DQN('MlpPolicy', env)
agent.learn(int(1e1))
agent.save('agents/dqn_highway')

# env = gym.make('highway_local-v0')
# config = {
#         "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
#     }
# agent = agent_factory(env, config)
# # implement deterministic greedy policy
# agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
#
# agent.learn(int(1e1))
# agent.save('agents/dqn_highway')