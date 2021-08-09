import argparse
from itertools import permutations
from os.path import abspath

from highway_disagreements.compare_agents import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-env', '--env_id', help='environment name', default="fastRight-v0")
    parser.add_argument('-a1', '--a1_name', help='agent name', type=str, default="Agent-1")
    parser.add_argument('-a2', '--a2_name', help='agent name', type=str, default="Agent-2")
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int,
                        default=3)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-l', '--horizon', help='number of frames to show per highlight',
                        type=int, default=10)
    parser.add_argument('-sb', '--show_score_bar', help='score bar', type=bool, default=False)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-k', '--n_disagreements', help='# of disagreements in the summary',
                        type=int, default=5)
    parser.add_argument('-overlaplim', '--similarity_limit', help='# overlaping',
                        type=int, default=3)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='trajectory')
    parser.add_argument('-impTraj', '--trajectory_importance',
                        help='method calculating trajectory importance', default='last_state')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='bety')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-ass', '--agent_assessment', help='apply agent ratio by agent score',
                        default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    parser.add_argument('-res', '--results_dir', help='results directory', default='results')
    parser.add_argument('-tr', '--traces_path', help='path to traces file if exists',
                        default=None)
    args = parser.parse_args()

    """get more/less trajectories"""
    # args.similarity_limit = 3  # int(args.horizon * 0.66)
    """importance measures"""
    args.importance = "last_state"
    # # traj: last_state, max_min, max_avg, avg, avg_delta
    # # state: sb, bety

    """"""
    args.verbose = False
    args.horizon = 20
    args.fps = 4
    args.num_episodes = 7
    args.randomized = True
    args.results_dir = abspath('../highway_disagreements/results')

    # agents = {
    #     "OnlySpeed": "run_20210729-160525_37180",
    #     "ClearLane": "run_20210729-160525_37182",
    #     "OnlySafe": 'run_20210729-160525_37184',
    #     "FastRight": 'run_20210729-160524_37188',
    #     "RightLane": 'run_20210729-160525_37181'
    # }
    agents = ["SocialDistance", "FastRight", "ClearLane"]
    """RUN"""
    for a1, a2 in permutations(agents, 2):
        args.a1_name = a1
        args.a2_name = a2
        args.a1_path = abspath(f'../agents/TheOne/{a1}')
        args.a2_path = abspath(f'../agents/TheOne/{a2}')
        main(args)
