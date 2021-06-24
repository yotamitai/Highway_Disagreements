import gym
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory


def get_agent(config, env=None, env_id=None, seed=None, offscreen_rendering=True):
    """Implement here for specific agent and environment loading scheme"""
    if not env:
        assert env_id, 'No env_id supplied for agent environment'
        assert seed is not None, 'No random seed supplied for agent environment'
        env = gym.make(env_id)
        env.seed(seed)
        env.configure({"offscreen_rendering": offscreen_rendering})
    # Make agent
    agent = agent_factory(env, config)
    # implement deterministic greedy policy
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    return env, agent

# def copy_env_agent(env, config, actions=[]):
#     agent = agent_factory(env, config)
#     # implement deterministic greedy policy
#     agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
#     for a in actions:
#         _ = env.step(a)
#     return agent

# def reload_agent(e, config, seed, actions):
#     env, agent = get_agent(config, seed)
#     # each episode resets the environment.
#     # each such env is different in randomly generated parts
#     # so -- to load the same game env there is need to do the same number of resets
#     [env.reset() for _ in range(e+1)]
#     #TODO make sure the resulting environment is truely the same as env1.
#     for a in actions:
#         _ = env.step(a)
#     return env, agent
