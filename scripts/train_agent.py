import gym
import highway_env
from scripts.experiments import evaluate
from stable_baselines import DQN


def configure_env():
    env = gym.make('highway-v0')
    env.configure({
        "lanes_count": 4,
        "vehicles_count": 40,
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 40,
    })
    env.reset()
    return env

env = configure_env()
agent = DQN('MlpPolicy', env)
agent.learn(int(1e4))
agent.save('agents/dqn_highway')