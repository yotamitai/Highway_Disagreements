import argparse
from highway_disagreements.get_agent import get_agent
from highway_disagreements.logging_info import log


def agent_assessment(a1_config, a2_config):
    agent_ratio, a1_overall, a2_overall = assess_agents(a1_config, a2_config)
    msg = f'A1 score: {a1_overall}, A2 score: {a2_overall}, agent_ration: {agent_ratio}'
    log(args.logger, msg, args.verbose)
    return agent_ratio

def assess_agents(a1, a2):
    a1_overall = agent_score(a1)
    a2_overall = agent_score(a2)
    if a1_overall < 0 < a2_overall:
        a2_overall += abs(a1_overall)
        a1_overall = 1
    if a2_overall < 0 < a1_overall:
        a1_overall += abs(a2_overall)
        a2_overall = 1
    return a1_overall / a2_overall, a1_overall, a2_overall


def agent_score(config):
    """implement a simulation of the agent and retrieve the in-game score"""
    env, agent, agent_args = get_agent(config)
    scores = []
    for k in range(5):
        curr_obs, rewards = env.reset(), []
        done = False
        while not done:
            a = agent.act(curr_obs)
            obs, r, done, infos = env.step(a)
            rewards.append(r)
        scores.append(sum(rewards))

    env.close()
    return sum(scores) / len(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='agent configuration')
    args = parser.parse_args()
    agent_score(args.config)