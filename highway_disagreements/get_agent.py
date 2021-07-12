from os.path import abspath
from pathlib import Path

import gym
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation


class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='../agents', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes,
                                           display_env=display_env)


def get_agent(config, env=None, env_id=None, seed=None, offscreen_rendering=True):
    """Implement here for specific agent and environment loading scheme"""
    if not env:
        assert env_id, 'No env_id supplied for agent environment'
        assert seed is not None, 'No random seed supplied for agent environment'
        env = gym.make(env_id)
        env.seed(seed)
        env.configure({"offscreen_rendering": offscreen_rendering})
    # config agent agent
    agent = agent_factory(env, config)
    # implement deterministic greedy policy
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    # create evaluation
    evaluation = MyEvaluation(env, agent, display_env=False)
    agent_path = Path(abspath(config['path']))
    # load agent
    evaluation.load_agent_model(agent_path)
    agent = evaluation.agent

    return env, agent
