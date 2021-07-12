from os.path import abspath
from pathlib import Path

import gym
import highway_env
from highway_disagreements.envs.highway_env_local import LocalHighwayEnv
from highway_disagreements.get_agent import MyEvaluation
from rl_agents.agents.common.factory import agent_factory


agent_config = {
        "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
        "path": 'agents/DQN_1000ep/checkpoint-final.tar',
    }

env = gym.make('highway_local-v0')
env.configure({"offscreen_rendering": True})
agent = agent_factory(env, agent_config)

"""train agent"""
evaluation = MyEvaluation(env, agent, output_dir='agents', num_episodes=10, display_env=False)
evaluation.train()

"""load agent"""
# evaluation = MyEvaluation(env, agent, num_episodes=3, display_env=True)
# agent_path = Path(abspath(agent_config['pretrained_model_path']))
# evaluation.load_agent_model(agent_path)


"""evaluate"""
# evaluation.test()

