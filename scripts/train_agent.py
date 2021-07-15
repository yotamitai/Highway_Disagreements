import json
from os.path import abspath
from pathlib import Path

import gym
import highway_env
from highway_disagreements.get_agent import MyEvaluation
from rl_agents.agents.common.factory import agent_factory


# agent_config = {
#         "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
#         "path": 'agents/DQN_1000ep/checkpoint-final.tar',
#     }
f = open(abspath('highway_disagreements/agent_configs/fastSafe.json'))
agent_config = json.load(f)
env = gym.make('highway_local-v0')
agent = agent_factory(env, agent_config)
env.configure(agent_config)

"""train agent"""
evaluation = MyEvaluation(env, agent, output_dir='agents', num_episodes=10, display_env=False)
evaluation.train()

"""load agent"""
# evaluation = MyEvaluation(env, agent, num_episodes=3, display_env=True, output_dir='agents')
# agent_path = Path(abspath('agents/LocalHighwayEnv/DQNAgent/run_20210715-134954_75081/checkpoint-final.tar'))
# evaluation.load_agent_model(agent_path)

"""evaluate"""
# evaluation.test()

