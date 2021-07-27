import argparse
import json
from os.path import abspath
from pathlib import Path

import gym
import highway_env
from highway_disagreements.get_agent import MyEvaluation
from rl_agents.agents.common.factory import agent_factory


def config(config_path):
    f = open(config_path)
    env_config = json.load(f)
    env = gym.make(env_config["env_id"])
    agent = agent_factory(env, env_config)
    env.configure(env_config)
    env.define_spaces()
    return env, agent

def train_agent(env, agent, num_episodes):
    """train agent"""
    evaluation = MyEvaluation(env, agent, output_dir='agents', num_episodes=num_episodes, display_env=False)
    evaluation.train()
    return evaluation


def load_agent(env, agent, load_path, num_episodes):
    """load agent"""
    evaluation = MyEvaluation(env, agent, num_episodes=num_episodes, display_env=True, output_dir='agents')
    agent_path = Path(load_path)
    evaluation.load_agent_model(agent_path)
    return evaluation

def test_agent(evaluation):
    evaluation.test()


def main(args):
    env, agent = config(args.config_path)
    evaluation = load_agent(env, agent, args.load_path, args.num_episodes) if args.load_path \
        else train_agent(env, agent, args.num_episodes)
    if args.eval: test_agent(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    # parser.add_argument('-env', '--env_id', help='environment name', default="highway_local-v0")
    parser.add_argument('-load', '--load_path', help='path to pre-trained agent', default=None)
    parser.add_argument('-config', '--config_path', help='path to env config file', default=None)
    parser.add_argument('-n_ep', '--num_episodes', help='number of episodes to run for test or train', default=3, type=int)
    parser.add_argument('-eval', '--eval', help='run evaluation', default=False)
    args = parser.parse_args()

    # args.load_path = abspath('agents/Saved_agents/clearLane_1000ep/checkpoint-final.tar')
    args.config_path = abspath('highway_disagreements/envs/env_configs/rightLane.json')
    args.eval = True
    args.num_episodes = 10

    main(args)