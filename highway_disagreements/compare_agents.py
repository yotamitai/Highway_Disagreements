import argparse
import json
import os
import random
from os.path import join, abspath
from pathlib import Path

import numpy as np
from numpy import argmax

from highway_disagreements.disagreement import save_disagreements, get_top_k_disagreements, \
    disagreement, \
    DisagreementTrace, State, make_same_length
from highway_disagreements.get_agent import get_agent
from highway_disagreements.get_trajectories import rank_trajectories
from highway_disagreements.mark_agents import get_and_mark_frames
from highway_disagreements.merge_and_fade import merge_and_fade
from highway_disagreements.side_by_side import side_by_side_video
from highway_disagreements.agent_score import agent_assessment
from highway_disagreements.logging_info import get_logging, log
from highway_disagreements.utils import load_traces, save_traces
from copy import deepcopy


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """get agents and environments"""
    env1, a1, evaluation1 = get_agent(args.a1_path)
    env2, a2, evaluation2 = get_agent(args.a2_path)
    args.logger.parent = None
    env2 = deepcopy(env1)
    env1.args = env2.args = args

    """agent assessment"""
    agent_ratio = 1
    # agent_ratio = 1 if not args.agent_assessment else \
    #     agent_assessment(args.a1_config, args.a2_config)

    """Run"""
    traces = []
    for e in range(args.num_episodes):
        log(args.logger, f'Running Episode number: {e}', args.verbose)
        trace = DisagreementTrace(e, args.horizon, agent_ratio)
        curr_obs, _ = env1.reset(), env2.reset()
        assert curr_obs.tolist() == _.tolist(), f'Nonidentical environment'
        a1.previous_state = a2.previous_state = curr_obs
        t, r, done = 0, 0, False
        """initial state"""
        curr_s = curr_obs
        a1_s_a_values = a1.get_state_action_values(curr_obs)
        a2_s_a_values = a2.get_state_action_values(curr_obs)
        frame = env1.render(mode='rgb_array')
        position = deepcopy(env1.road.vehicles[0].destination)
        state = State(t, e, curr_obs, curr_s, a1_s_a_values, frame, position)
        a1_a, a2_a = a1.act(curr_s), a2.act(curr_s)
        trace.update(state, curr_obs, a1_a, a1_s_a_values, a2_s_a_values, 0, False, {}, position)
        while not done:
        # for _ in range(50):
        #     if done: break
            """check for disagreement"""
            if a1_a != a2_a:
                log(args.logger, f'\tDisagreement at step {t}:\t\t A1: {a1_a} Vs. A2: {a2_a}',
                    args.verbose)
                copy_env2 = deepcopy(env2)
                disagreement(t, trace, env2, a2, curr_s, a1)
                """return agent 2 to the disagreement state"""
                env2 = copy_env2
                a2.previous_state = a1.previous_state
                # disagreement(t, trace, env2, a2, curr_s, a1)
                # """return agent 2 to the disagreement state"""
                # env2, a2, evaluation2 = get_agent(args.a2_path, evaluation_reset=evaluation2)
                # env2.args = args
                # init_state = [env2.reset() for _ in range(e + 1)][-1]
                # a2.previous_state = init_state
                # [env2.step(a) for a in trace.actions[:-1]]
                # assert a1.previous_state.tolist() == a2.previous_state.tolist(), \
                #     f'Nonidentical agent transition'
            """Transition both agent's based on agent 1 action"""
            t += 1
            curr_obs, r, done, info = env1.step(a1_a)
            position = deepcopy(env1.road.vehicles[0].destination)
            _ = env2.step(a1_a)  # dont need returned values
            assert curr_obs.tolist() == _[0].tolist(), f'Nonidentical environment transition'
            curr_s = curr_obs
            a1_s_a_values = a1.get_state_action_values(curr_obs)
            a2_s_a_values = a2.get_state_action_values(curr_obs)
            frame = env1.render(mode='rgb_array')
            state = State(t, e, curr_obs, curr_s, a1_s_a_values, frame, position)
            a1_a, a2_a = a1.act(curr_s), a2.act(curr_s)
            trace.update(state, curr_obs, a1_a, a1_s_a_values, a2_s_a_values, 0, False, info, position)

        """end of episode"""
        # trace.get_trajectories()
        traces.append(deepcopy(trace))

    """close environments"""
    env1.close()
    env2.close()
    evaluation1.close()
    evaluation2.close()
    return traces


def main(args):
    name, args.logger = get_logging(args)
    traces = load_traces(args.traces_path) if args.traces_path else online_comparison(args)
    log(args.logger, f'Obtained traces', args.verbose)

    """save traces"""
    save_traces(traces, args.output)
    log(args.logger, f'Saved traces', args.verbose)

    """get trajectories"""
    [trace.get_trajectories() for trace in traces]
    log(args.logger, f'Obtained trajectories', args.verbose)

    """rank disagreement trajectories by importance measures"""
    rank_trajectories(traces, args.importance)
    log(args.logger, f'Trajectories ranked', args.verbose)


    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    if not disagreements:
        log(args.logger, f'No disagreements found', args.verbose)
        return
    log(args.logger, f'Obtained {len(disagreements)} disagreements', args.verbose)

    """make all trajectories the same length"""
    disagreements = make_same_length(disagreements, args.horizon, traces)

    """randomize order"""
    if args.randomized: random.shuffle(disagreements)

    """mark disagreement frames"""
    a1_disagreement_frames, a2_disagreement_frames = \
        get_and_mark_frames(disagreements, traces, agent_position=[164, 66], color=args.color)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames,
                                   args.output, args.fps)
    log(args.logger, f'Disagreements saved', args.verbose)

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration + 11  # +11 from pause in save_disagreements
    # side_by_side_video(video_dir, args.n_disagreements, fade_out_frame, name)
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame, name)
    log(args.logger, f'DAs Video Generated', args.verbose)

    """ writes results to files"""
    log(args.logger, f'\nResults written to:\n\t\'{args.output}\'', args.verbose)


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
    parser.add_argument('-imp', '--importance',
                        help='importance method', default='last_state')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-ass', '--agent_assessment', help='apply agent ratio by agent score',
                        default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    parser.add_argument('-res', '--results_dir', help='results directory',
                        default=abspath('results'))
    parser.add_argument('-tr', '--traces_path', help='path to traces file if exists',
                        default=None)
    args = parser.parse_args()

    """get more/less trajectories"""
    # args.similarity_limit = 3  # int(args.horizon * 0.66)
    """importance"""
    args.importance = "last_state"
    # traj: last_state, max_min, max_avg, avg, avg_delta
    # state: sb, bety

    """"""
    args.verbose = False
    args.horizon = 30
    args.fps = 4
    args.num_episodes = 1
    args.randomized = True
    # args.n_disagreements = 2

    args.color = 0
    args.a1_name = 'SocialDistance'
    args.a2_name = 'ClearLane'
    args.a1_path = f'../agents/TheBest/{args.a1_name}'
    args.a2_path = f'../agents/TheBest/{args.a2_name}'

    # args.traces_path = join(abspath('results'),"2021-08-22_12:18:57_ParallelDriver-NoLaneChange")
    args.traces_path = '../User Study Videos/DA/TheBest/Current/2021-08-21_10:01:17_SocialDistance-ClearLane'

    """RUN"""
    main(args)

    # base_dir = '/home/yotama/OneDrive/Local_Git/Highway_Disagreements/User Study Videos/DA/DA_trajLastState_bety_7fps_20frames'
    # directories = os.listdir(base_dir)
    # directories = [x for x in directories if 'Parallel' not in x]
    #
    # """RUN"""
    # parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    #
    # for d in directories:
    #     f = open(Path(join(base_dir, d, 'metadata.json')))
    #     t_args = argparse.Namespace()
    #     t_args.__dict__.update(json.load(f))
    #     args = parser.parse_args(namespace=t_args)
    #     args.traces_path = join(base_dir, d)
    #     args.color = 255
    #
    #     """get more/less trajectories"""
    #     # args.similarity_limit = 3  # int(args.horizon * 0.66)
    #     """importance measures"""
    #     args.importance = "max_min"
    #     # # traj: last_state, max_min, max_avg, avg, avg_delta
    #     # # state: sb, bety
    #     args.verbose = True
    #     args.fps = 4
    #     args.randomized = True
    #     args.n_disagreements = 5
    #     main(args)
