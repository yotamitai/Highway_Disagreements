import argparse
import json
from datetime import datetime
import random
from pathlib import Path

import gym
import pandas as pd
from os.path import join, basename, abspath

from highway_disagreements.get_agent import get_agent
from get_traces import get_traces
from highlights.utils import create_video, make_clean_dirs, pickle_save, pickle_load
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images
from ffmpeg import merge_and_fade


def get_highlights(args):
    args.output_dir = join(abspath('results'), '_'.join(
        [datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_'), args.name]))
    make_clean_dirs(args.output_dir)
    with Path(join(args.output_dir,'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.load_dir:
        """Load traces and state dictionary"""
        traces = pickle_load(join(args.load_dir, 'Traces.pkl'))
        states = pickle_load(join(args.load_dir, 'States.pkl'))
        if args.verbose: print(f"Highlights {15 * '-' + '>'} Traces & States Loaded")
    else:
        env, agent, _ = get_agent(args.load_path)
        env.args = args
        traces, states = get_traces(env, agent, args)

    """Save data used for this run in output dir"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))

    """highlights algorithm"""
    a, b, c = states[(0, 0)].image.shape
    data = {'state': list(states.keys()),
            'q_values': [x.observed_actions for x in states.values()],
            'features': [x.image.reshape(a * b * c) for x in states.values()]}


    if args.highlights_div:
        i = len(traces[0].states) // 2
        threshold = args.div_coefficient * (
        sum(states[(0, i)].image.reshape(a * b * c) - states[(0, i + 1)].image.reshape(
            a * b * c)))
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    highlights_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        trace_lengths = {k: len(v.states) for k, v in enumerate(traces)}
        if args.highlights_div:
            summary_states = highlights_div(highlights_df, trace_lengths, args.num_trajectories,
                                            args.trajectory_length, args.minimum_gap,
                                            threshold=threshold)
        else:
            summary_states = highlights(highlights_df, trace_lengths, args.num_trajectories,
                                        args.trajectory_length, args.minimum_gap)
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

    """Save trajectories used for this run in output dir"""
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

    if not args.load_dir: env.close()
    # del gym.configs.registration.registry.env_specs[env.spec.id]


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
    parser.add_argument('--highlights_div', type=bool, default=False)
    parser.add_argument('--div_coefficient', type=int, default=2)
    args = parser.parse_args()

    """Highlight parameters"""
    args.n_traces = 3
    args.trajectory_importance = "single_state" # single_state
    args.state_importance = "second"
    args.num_trajectories = 5
    args.trajectory_length = 30
    args.fade_duration = 2
    args.minimum_gap = 0
    args.overlay_limit = 5
    args.allowed_similar_states = 3
    args.highlights_selection_method = 'importance_scores'  # 'scores_and_similarity', 'similarity'
    args.randomized = True
    args.fps = 4
    args.verbose = True
    args.load_trajectories = False
    args.results_dir = abspath('results')
    # args.highlights_div = True
    # args.div_coefficient = 2

    # RUN
    # agent = 'FastRight'
    # args.load_dir = '/home/yotama/OneDrive/Local_Git/Highway_Disagreements/highlights/results/2021-08-09_10:29:58_FastRight'
    # args.load_path = f'results/ClearLane_09:48:40_04-08-2021'
    # args.agent_path = f'../agents/TheOne/{agent}/checkpoint-best.tar'
    # args.env_config = abspath(f'../highway_disagreements/configs/env_configs/{agent}.json')
    # args.env_id = "fastRight-v0"
    # args.seed = 0
    # args.name = agent

    args.load_dir = False
    args.name = 'ParallelDriver'
    args.load_path = f'../agents/TheBest/{args.name}'
    get_highlights(args)
