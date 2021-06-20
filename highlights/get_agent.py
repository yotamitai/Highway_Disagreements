from rl_agents.agents.common.factory import agent_factory
from envs.highway_env_local import LocalHighwayEnv

def get_agent(args):
    """Implement here for specific agent and environment loading scheme"""
    # env = gym.make("highway-v0")
    env = LocalHighwayEnv()
    # Make agent
    agent = agent_factory(env, args.agent_config)

    return env, agent, {}

"""Actions:"""
# 0: 'LANE_LEFT',
# 1: 'IDLE',
# 2: 'LANE_RIGHT',
# 3: 'FASTER',
# 4: 'SLOWER'
