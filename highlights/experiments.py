import argparse
from get_highlights import get_highlights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int,
                        default=100)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-a', '--agent_dir', help='directory from which to load the agent')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console',
                        action='store_true')
    args = parser.parse_args()

    """agent parameters"""
    args.results_dir = '/home/yotama/Local_Git/InterestingnessXRL/Highlights/results'
    args.num_episodes = 1  # max 2000 (defined in configuration.py)
    args.fps = 2
    args.verbose = True
    args.record = 'all'
    args.show_score_bar = False
    args.clear_results = True

    """Highlight parameters"""
    args.n_traces = 3
    args.trajectory_importance = "single_state"
    args.state_importance = "second"
    args.num_trajectories = 5
    args.trajectory_length = 10
    args.fade_duration = 2
    args.minimum_gap = 0
    args.overlay_limit = 3
    args.allowed_similar_states = 3
    args.highlights_selection_method = 'importance_scores'  # 'scores_and_similarity', 'similarity'
    args.load_traces = False
    args.load_trajectories = False
    args.randomized = True

    """Experiments"""
    for agent in ['Expert', 'LimitedVision', 'HighVision', 'Novice', 'FearWater']:
        args.agent_dir = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/' + agent
        args.name = agent
        get_highlights(args)