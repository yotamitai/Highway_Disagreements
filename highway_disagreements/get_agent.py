import json
from os.path import abspath
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


def get_agent(trained_agent_path, env=None, env_config=None, env_id=None, seed=None, args=None):
    """Implement here for specific agent and environment loading scheme"""
    f = open(env_config)
    env_config = json.load(f)
    if not env:
        assert env_id, 'No env_id supplied for agent environment'
        assert seed is not None, 'No random seed supplied for agent environment'
        env = gym.make(env_id)
        env.configure(env_config)
    # config agent agent
    agent = agent_factory(env, env_config)
    # implement deterministic greedy policy
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    # create evaluation
    evaluation = MyEvaluation(env, agent, display_env=False)
    agent_path = Path(abspath(trained_agent_path))
    # load agent
    evaluation.load_agent_model(agent_path)
    agent = evaluation.agent
    if args: env.args = args
    return env, agent
