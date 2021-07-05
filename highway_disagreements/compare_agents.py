import argparse
import random
from os.path import abspath
from os.path import join
from disagreement import save_disagreements, get_top_k_disagreements, disagreement, \
    DisagreementTrace, State
from get_agent import get_agent
from highway_disagreements.agent_score import agent_assessment
from highway_disagreements.logging_info import get_logging, log
from highway_disagreements.side_by_side import side_by_side_video
from highway_disagreements.utils import load_traces, save_traces
from copy import deepcopy


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """get agents and environments"""
    env1, a1 = get_agent(args.a1_config, env_id=args.env_id, seed=args.seed)
    env1.args = args
    env2, a2 = get_agent(args.a2_config, env=deepcopy(env1))

    """agent assessment"""
    agent_ratio = 1 if not args.agent_assessment else \
        agent_assessment(args.a1_config, args.a2_config)

    """Run"""
    traces = []
    for e in range(args.num_episodes):
        log(f'Running Episode number: {e}', args.verbose)
        curr_obs, _ = env1.reset(), env2.reset()
        a1.previous_state = curr_obs
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


def rank_trajectories(traces, importance_type, state_importance, traj_importance):
    # TODO check that trajectories are being summed correctly (different lengths)
    # check that
    for trace in traces:
        for i, trajectory in enumerate(trace.disagreement_trajectories):
            if importance_type == 'state':
                importance = trajectory.calculate_state_importance(state_importance)
            else:
                # TODO check that all importance criteria work
                importance = trajectory.calculate_trajectory_importance(trace, i, traj_importance,
                                                                        state_importance)
            trajectory.importance = importance


def main(args):
    name, file_name = get_logging(args)
    traces = load_traces(args.traces_path) if args.traces_path else online_comparison(args)
    log(f'Obtained traces', args.verbose)

    """save traces"""
    # TODO check if json works better than pickle
    output_dir = join(args.results_dir, file_name)
    save_traces(traces, output_dir)
    log(f'Saved traces', args.verbose)

    """rank disagreement trajectories by importance measures"""
    rank_trajectories(traces, args.importance_type, args.state_importance,
                      args.trajectory_importance)

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    log(f'Obtained {len(disagreements)} disagreements', args.verbose)

    """randomize order"""
    if args.randomized: random.shuffle(disagreements)

    """mark disagreement frames"""
    # TODO mark agent, needed?
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        # state_idx = d.a1_states[(args.horizon // 2) - 1]
        # state = traces[d.episode].states[state_idx]
        # state.image = mark_agent(state.image, state.position)
        a1_frames, a2_frames = traces[d.episode].get_frames(d.a1_states, d.a2_states,
                                                            d.trajectory_index,
                                                            mark_position=[164, 66])
        # for i in range(args.horizon // 2, args.horizon):
        #     a1_position = d.a1_states[i].agent_position
        #     a2_position = d.a2_states[i].agent_position
        #     a1_frames[i] = mark_agent(a1_frames[i], position=a1_position, color=(255, 0,))
        #     a2_frames[i] = mark_agent(a2_frames[i], position=a2_position, color=(0, 0, 0))
        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames, output_dir,
                                   args.fps)
    log(f'Disagreements saved', args.verbose)

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration
    side_by_side_video(video_dir, args.n_disagreements, fade_out_frame, name)
    log(f'DAs Video Generated', args.verbose)

    """ writes results to files"""
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
    parser.add_argument('-ass', '--agent_assessment', help='apply agent ratio by agent score',
                        default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    parser.add_argument('-res', '--results_dir', help='results directory', default='results')
    parser.add_argument('-tr', '--traces_path', help='path to traces file if exists',
                        default=None)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_config = {
        "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
        "pretrained_model_path": '../agents/DQN_1000ep/checkpoint-final.tar',
    }
    args.a2_config = {
        "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
        "pretrained_model_path": '../agents/DQN_10ep/checkpoint-final.tar',
    }

    # args.fps = 1

    # args.show_score_bar = False
    # args.n_disagreements = 5
    # args.randomized = True
    """get more/less trajectories"""
    # args.similarity_limit = 3  # int(args.horizon * 0.66)
    """importance measures"""
    args.state_importance = "bety"  # "sb" "bety"
    args.trajectory_importance = "avg"  # last_state, max_min, max_avg, avg, avg_delta
    args.importance_type = 'trajectory'  # state/trajectory

    """"""
    args.verbose = False
    args.horizon = 20
    args.fps = 5
    args.num_episodes = 3
    # args.a1_name = args.a1_config["__class__"].split('.')[-1][:-2]
    # args.a2_name = args.a2_config["__class__"].split('.')[-1][:-2]
    args.a1_name = args.a1_config["pretrained_model_path"].split('/')[2]
    args.a2_name = args.a2_config["pretrained_model_path"].split('/')[2]
    args.results_dir = abspath('results')
    # args.traces_path = join('results', 'DQNAgent_FTQAgent_29-06_10:42:33')
    args.traces_path = None

    """RUN"""
    main(args)
