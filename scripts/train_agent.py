import argparse
import json
from os import listdir
from os.path import abspath, join
from pathlib import Path

import gym
import highway_env
from highway_disagreements.get_agent import MyEvaluation
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory


def config(env_config, agent_config):
    env = gym.make(env_config["env_id"])
    env.configure(env_config)
    env.define_spaces()
    agent = agent_factory(env, agent_config)
    return env, agent


def train_agent(env_config_path, agent_config_path, num_episodes):
    """train agent"""
    f1, f2 = open(env_config_path), open(agent_config_path)
    env_config, agent_config = json.load(f1), json.load(f2)
    env, agent = config(env_config, agent_config)
    evaluation = MyEvaluation(env, agent, output_dir='agents', num_episodes=num_episodes,
                              display_env=False)
    evaluation.train()
    return evaluation


def load_agent(load_path, num_episodes):
    """load agent"""
    config_filename = [x for x in listdir(load_path) if "metadata" in x][0]
    f = open(join(load_path, config_filename))
    config_dict = json.load(f)
    env_config, agent_config, = config_dict['env'], config_dict['agent']
    env, agent = config(env_config, agent_config)
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    evaluation = MyEvaluation(env, agent, num_episodes=num_episodes, display_env=True,
                              output_dir='agents')
    agent_path = Path(join(load_path, 'checkpoint-final.tar'))
    evaluation.load_agent_model(agent_path)
    return evaluation


def test_agent(evaluation):
    evaluation.test()


def main(args):
    evaluation = load_agent(args.load_path, args.num_episodes) if args.load_path \
        else train_agent(args.env_config, args.agent_config, args.num_episodes)
    if args.eval: test_agent(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    # parser.add_argument('-env', '--env_id', help='environment name', default="highway_local-v0")
    parser.add_argument('-load', '--load_path', help='path to pre-trained agent', default=None)
    parser.add_argument('-a_cnfg', '--agent_config', help='path to env config file', default=None)
    parser.add_argument('-e_cnfg', '--env_config', help='path to env config file', default=None)
    parser.add_argument('-n_ep', '--num_episodes',
                        help='number of episodes to run for test or train', default=3, type=int)
    parser.add_argument('-eval', '--eval', help='run evaluation', default=False)
    args = parser.parse_args()


    # env_config = "FastRight"
    # agent_config = "ddqn"
    # args.agent_config = abspath(f'highway_disagreements/configs/agent_conefigs/{agent_config}.json')
    # args.env_config = abspath(f'highway_disagreements/configs/env_configs/{env_config}.json')

    # args.load_path = '../agents/Yael'
    # args.eval = True
    # #
    # args.num_episodes = 4

    main(args)
