import argparse
from datetime import datetime
import random

import gym
import pandas as pd
from os.path import join, basename

from get_agent import get_agent
from get_traces import get_traces
from utils import create_video, make_clean_dirs, pickle_save
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images
from ffmpeg import merge_and_fade


def get_highlights(args):
    args.output_dir = join(args.results_dir, '_'.join(
        [basename(args.agent_dir), datetime.now().strftime("%H:%M:%S_%d-%m-%Y")]))
    make_clean_dirs(args.output_dir)

    env, agent, agent_args = get_agent(args)
    [env.reset() for _ in range(5)]
    traces, states = get_traces(env, agent, agent_args, args)

    """highlights algorithm"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(highlights_df, traces, args.num_trajectories,
                                    args.trajectory_length, args.minimum_gap, args.overlay_limit)
        # summary_states = highlights_div(highlights_df, traces, args.num_trajectories,
        #                             args.trajectory_length,
        #                             args.minimum_gap)
        all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
        summary_trajectories = all_trajectories

    else:
        """highlights importance by trajectory"""
        all_trajectories, summary_trajectories = \
            trajectories_by_importance(traces, state_importance_dict, args)

    # random highlights
    # summary_trajectories = random.choices(all_trajectories, k=5)

    # random order
    if args.randomized: random.shuffle(summary_trajectories)

    """Save data used for this run"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))

    """Save Highlight videos"""
    frames_dir = join(args.output_dir, 'Highlight_Frames')
    videos_dir = join(args.output_dir, "Highlight_Videos")
    height, width, layers = list(states.values())[0].image.shape
    img_size = (width, height)
    get_trajectory_images(summary_trajectories, states, frames_dir)
    create_video(frames_dir, videos_dir, args.num_trajectories, img_size, args.fps)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Videos Generated")

    """Merge Highlights to a single video with fade in/ fade out effects"""
    fade_out_frame = args.trajectory_length - args.fade_duration
    merge_and_fade(videos_dir, args.num_trajectories, fade_out_frame, args.fade_duration,
                   args.name)

    """Save data used for this run"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))
    if args.verbose: print(f"Highlights {15 * '-' + '>'} Run Configurations Saved")

    env.close()
    del gym.envs.registration.registry.env_specs[env.spec.id]


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
    args.agent_dir = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Expert_Mid'
    args.results_dir = '/home/yotama/Local_Git/InterestingnessXRL/Highlights/results'
    args.num_episodes = 1  # max 2000 (defined in configuration.py)
    args.fps = 2
    args.verbose = True
    args.record = 'all'
    args.show_score_bar = False
    args.clear_results = True

    """Highlight parameters"""
    args.n_traces = 10
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

    # RUN
    args.name = basename(args.agent_dir)
    get_highlights(args)
