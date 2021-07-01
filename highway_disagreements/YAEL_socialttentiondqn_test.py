
import sys

import gym
import highway_env
from rl_agents.agents.common.factory import agent_factory

sys.path.insert(0,'/data/home/yael123/highway/highway-env/scripts/')
rl_agents_dir = '/data/home/yael123/highway/rl-agents/'
sys.path.append(rl_agents_dir)

# from utils import show_videos

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

# os.chdir(rl_agents_dir + "/scripts/")
# env_config = 'configs/HighwayEnv/env.json'
# agent_config = 'configs/HighwayEnv/agents/DQNAgent/ddqn.json'

env = gym.make('highway-v0')
#env.config["lanes_count"] = 4
#env.config["vehicles_count"]=30
#env.config["vehicles_density"] = 2
#env.reset()

agent_config = {
        # "__class__": "<class 'rl_agents.agents.simple.open_loop.OpenLoopAgent'>",
        "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
        "gamma": 0.7,
    }
agent = agent_factory(env, agent_config)

evaluation = Evaluation(env, agent, num_episodes=3000, display_env=False)
print("NO TRAIN 26/04/2021")
print("6 lanes")
print(f"Ready to train {agent} on {env}")

"""Run tensorboard locally to visualize training."""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir "{evaluation.directory}"

"""Start training. This should take about an hour."""

evaluation.train()

"""Progress can be visualised in the tensorboard cell above, which should update every 30s (or manually). You may need to click the *Fit domain to data* buttons below each graph.

## Testing

Run the learned policy for a few episodes.
"""

env = load_environment(env_config)
env.configure({"offscreen_rendering": True})
#env.config["lanes_count"] = 4
#env.config["vehicles_count"]=30
#env.config["vehicles_density"] = 2
#env.reset()
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=3000, recover=True)
evaluation.test()
show_videos(evaluation.run_directory)
