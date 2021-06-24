import argparse
import logging
import random
import os
from os.path import abspath
import gym
from datetime import datetime
from os.path import join, basename
from agent_score import asses_agents
from disagreement import save_disagreements, get_top_k_disagreements, disagreement, \
    DisagreementTrace, State
from get_agent import get_agent
from merge_and_fade import merge_and_fade
from highway_disagreements.utils import mark_agent, pickle_load, pickle_save, make_clean_dirs
from copy import deepcopy


def get_logging(args):
    if not os.path.exists(abspath('logs')):
        os.makedirs('logs')
    name = '_'.join([basename(args.a1_name), basename(args.a2_name)])
    file_name = '_'.join([name, datetime.now().strftime("%d-%m %H:%M:%S").replace(' ', '_')])
    log_name = join('logs', file_name)
    args.output = join('results', file_name)
    logging.basicConfig(filename=log_name + '.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    log(f'Comparing Agents: {name}', args.verbose)
    log(f'Disagreement importance by: {args.importance_type}', args.verbose)
    return name, file_name


def agent_assesment(a1_config, a2_config):
    agent_ratio, a1_overall, a2_overall = asses_agents(a1_config, a2_config)
    msg = f'A1 score: {a1_overall}, A2 score: {a2_overall}, agent_ration: {agent_ratio}'
    log(msg, args.verbose)
    return agent_ratio


def get_disagreement_traces(args):
    pass


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """get agents and environments"""
    env1, a1 = get_agent(args.a1_config, env_id=args.env_id, seed=args.seed)
    env1.args = args
    env2, a2 = get_agent(args.a2_config, env=deepcopy(env1))

    """Run"""
    traces = []
    for e in range(args.num_episodes):
        log(f'Running Episode number: {e}', args.verbose)
        curr_obs, _ = env1.reset(), env2.reset()
        """agent assessment"""
        agent_ratio = 1 if not args.agent_assesment \
            else agent_assesment(args.a1_config, args.a2_config)
        """get initial state"""
        t = 0
        done = False
        curr_s = curr_obs
        a1_s_a_values = a1.get_state_action_values(curr_obs)
        a2_s_a_values = a2.get_state_action_values(curr_obs)
        frame = env1.render(mode='rgb_array')
        position = env1.vehicle.position
        state = State(t, e, curr_obs, curr_s, a1_s_a_values, frame, position)
        a1_a, _ = a1.act(curr_s), a2.act(curr_s)
        """initiate and update trace"""
        trace = DisagreementTrace(e, args.horizon, agent_ratio)
        trace.update(state, curr_obs, a1_a, a1_s_a_values, a2_s_a_values, 0, False, {})
        while not done:
            """Observe both agent's desired action"""
            a1_a = a1.act(curr_s)
            a2_a = a2.act(curr_s)
            """check for disagreement"""
            if a1_a != a2_a:
                # if True:
                copy_env2 = deepcopy(env2)
                log(f'\tDisagreement at step {t}', args.verbose)
                disagreement(t, trace, env2, a2, curr_s, a1)
                """return agent 2 to the disagreement state"""
                env2 = copy_env2
            """Transition both agent's based on agent 1 action"""
            new_obs, r, done, info = env1.step(a1_a)
            _ = env2.step(a1_a)  # dont need returned values
            new_s = new_obs
            """get new state"""
            t += 1
            new_a1_s_a_values = a1.get_state_action_values(new_s)
            new_a2_s_a_values = a2.get_state_action_values(new_s)
            new_frame = env1.render(mode='rgb_array')
            new_position = env1.vehicle.position
            new_state = State(t, e, new_obs, new_s, new_a1_s_a_values, new_frame, new_position)
            new_a = a1.act(curr_s)
            """update trace"""
            trace.update(new_state, new_obs, new_a, new_a1_s_a_values,
                         new_a2_s_a_values, r, done, info)
            """update params for next iteration"""
            curr_s = new_s
        """end of episode"""
        trace.get_trajectories()
        traces.append(deepcopy(trace))

    """close environments"""
    env1.close()
    env2.close()
    return traces


def load_traces(path):
    return pickle_load(join(path, 'Traces.pkl'))


def save_traces(traces, output_dir):
    os.makedirs(output_dir)
    pickle_save(traces, join(output_dir, 'Traces.pkl'))


def rank_trajectories(traces, importance_type, state_importance, traj_importance):
    for trace in traces:
        for i, trajectory in enumerate(trace.disagreement_trajectories):
            if importance_type == 'state':
                trajectory.state_importance = state_importance
                importance = trajectory.calculate_state_importance(state_importance, a1_states[da_index],
                                                      a2_states[da_index],
                                                      disagreement_importance, agent_ratio)
            else:
                s_i, e_i = min(trajectory.a1_states), max(trajectory.a1_states)+1
                importance = trajectory.calculate_trajectory_importance(traj_importance, state_importance, trace.states[s_i: e_i],
                                                           trace.a2_trajectories[i])
            trajectory.importance = importance
            #TODO check that all importance criteria work


def log(msg, verbose):
    if verbose: print(msg)
    logging.info(msg)


def main(args):
    name, file_name = get_logging(args)
    traces = load_traces(args.traces_path) if args.traces_path else online_comparison(args)
    log(f'Obtained traces', args.verbose)

    """save traces"""
    # TODO
    output_dir = join(args.results_dir, file_name)
    save_traces(traces, output_dir)
    log(f'Saved traces', args.verbose)

    # TODO a1 and a2 disagreement trajectories are not of same length. take in consideration
    """rank disagreement trajectories by importance measures"""
    rank_trajectories(traces, args.importance_type, args.state_importance,
                      args.trajectory_importance)

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    log(f'Obtained {len(disagreements)} disagreements', args.verbose)

    """randomize order"""
    if args.randomized: random.shuffle(disagreements)

    """mark disagreement frames"""
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        d_state = d.a1_states[(args.horizon // 2) - 1]
        d_state.image = mark_agent(d_state.image, d_state.agent_position)
        a1_frames, a2_frames = d.get_frames()
        for i in range(args.horizon // 2, args.horizon):
            a1_position = d.a1_states[i].agent_position
            a2_position = d.a2_states[i].agent_position
            a1_frames[i] = mark_agent(a1_frames[i], position=a1_position, color=(255, 0,))
            a2_frames[i] = mark_agent(a2_frames[i], position=a2_position, color=(0, 0, 0))
        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames,
                                   args.output_dir, args.fps)
    log(f'Disagreements saved', args.verbose)

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame, name)
    log(f'DAs Video Generated', args.verbose)

    """ writes results to files"""
    a1.save(output_dir)
    log(f'\nResults written to:\n\t\'{output_dir}\'', args.verbose)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-env', '--env_id', help='environment name', default="highway_local-v0")
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
    parser.add_argument('-ass', '--agent_assesment', help='apply agent ratio by agent score',
                        default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_config = {
        "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    }
    args.a2_config = {
        "__class__": "<class 'rl_agents.agents.fitted_q.pytorch.FTQAgent'>",
    }

    # args.fps = 1
    # args.horizon = 10
    # args.show_score_bar = False
    # args.n_disagreements = 5
    # args.randomized = True
    """get more/less trajectories"""
    # args.similarity_limit = 3  # int(args.horizon * 0.66)
    """importance measures"""
    # args.state_importance = "bety"  # "sb" "bety"
    # args.trajectory_importance = "last_state" # last_state, max_min, max_avg, avg, avg_delta
    # args.importance_type = 'trajectory'  # state/trajectory

    """run params"""
    args.trajectory_importance = "max_min" # last_state, max_min, max_avg, avg, avg_delta
    args.num_episodes = 1
    args.a1_name = args.a1_config["__class__"].split('.')[-1][:-2]
    args.a2_name = args.a2_config["__class__"].split('.')[-1][:-2]
    args.results_dir = abspath('results')
    args.traces_path = join('results', 'DQNAgent_FTQAgent_24-06_14:43:27')

    """RUN"""
    main(args)
