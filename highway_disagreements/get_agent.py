import json
from os import listdir
from os.path import abspath, join
from pathlib import Path

import gym
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation


ACTION_DICT = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}

class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='../agents', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes,
                                           display_env=display_env)


def get_agent(load_path, seed=0):
    """Implement here for specific agent and environment loading scheme"""
    config_filename = [x for x in listdir(load_path) if "metadata" in x][0]
    f = open(join(load_path, config_filename))
    config_dict = json.load(f)
    env_config, agent_config, = config_dict['env'], config_dict['agent']
    env = gym.make(env_config["env_id"])
    env.seed(seed)
    agent = agent_factory(env, agent_config)
    env_config.update({"simulation_frequency": 15, "policy_frequency": 5, })
    env.configure(env_config)
    env.define_spaces()
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    evaluation = MyEvaluation(env, agent, display_env=False)
    agent_path = Path(join(load_path, 'checkpoint-final.tar'))
    evaluation.load_agent_model(agent_path)
    return env, agent, evaluation
