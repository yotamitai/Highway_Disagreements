import argparse
from os.path import abspath

from highlights.get_highlights import get_highlights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS')
    parser.add_argument('-a', '--name', help='agent name', type=str, default="Agent-0")
    parser.add_argument('-num_ep', '--num_episodes', help='number of episodes to run', type=int,
                        default=1)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-n', '--n_traces', help='number of traces to obtain', type=int,
                        default=10)
    parser.add_argument('-k', '--num_trajectories',
                        help='number of highlights trajectories to obtain', type=int, default=5)
    parser.add_argument('-l', '--trajectory_length',
                        help='length of highlights trajectories ', type=int, default=10)
    parser.add_argument('-v', '--verbose', help='print information to the console',
                        action='store_true')
    parser.add_argument('-overlapLim', '--overlay_limit', help='# overlaping', type=int,
                        default=3)
    parser.add_argument('-minGap', '--minimum_gap', help='minimum gap between trajectories',
                        type=int, default=0)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='single_state')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='second')
    parser.add_argument('-loadTrace', '--load_last_traces',
                        help='load previously generated traces', type=bool, default=False)
    parser.add_argument('-loadTraj', '--load_last_trajectories',
                        help='load previously generated trajectories', type=bool, default=False)
    args = parser.parse_args()

    """Highlight parameters"""
    args.fade_duration = 2
    args.minimum_gap = 0
    args.overlay_limit = 5
    args.allowed_similar_states = 3
    args.randomized = False
    args.verbose = True
    args.load_dir = False #abspath('results/safe_10:39:01_21-07-2021')
    args.load_trajectories = False
    args.results_dir = abspath('results')

    # RUN
    # agents = {
    #     "FastRight": "run_20210729-160525_37180",
    #     "ClearLane": "run_20210729-160525_37182",
    #     "OnlySafe": 'run_20210729-160525_37184',
    #     "FastRight": 'run_20210729-160524_37188',
    #     "RightLane": 'run_20210729-160525_37181'
    # }
    args.n_traces = 30
    args.num_trajectories = 5
    args.trajectory_length = 20
    args.fps = 4
    args.trajectory_importance = "single_state"  # single_state
    args.state_importance = "second"
    args.highlights_selection_method = 'importance_scores'  # 'scores_and_similarity', 'similarity'

    for agent in ["SocialDistance", "NoLaneChange", "ClearLane"]:
        args.name = agent
        args.load_path = f'../agents/TheOne/{agent}'
        get_highlights(args)

