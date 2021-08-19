from copy import deepcopy
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from numpy import asarray
from highway_disagreements.get_trajectories import trajectory_importance_max_min, \
    trajectory_importance_max_avg, trajectory_importance_avg, trajectory_importance_avg_delta
from highway_disagreements.get_agent import ACTION_DICT
from highway_disagreements.logging_info import log
from highway_disagreements.utils import save_image, create_video, make_clean_dirs
import imageio


class DisagreementTrace(object):
    def __init__(self, episode, trajectory_length, agent_ratio=1):
        self.positions = []
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.length = 0
        self.states = []
        self.trajectory_length = trajectory_length
        self.a2_trajectories = []
        self.a1_trajectory_indexes = []
        self.episode = episode
        self.a2_rewards = []
        self.a1_max_q_val = 0
        self.a2_max_q_val = 0
        self.a1_min_q_val = float('inf')
        self.a2_min_q_val = float('inf')
        self.agent_ratio = agent_ratio
        self.disagreement_indexes = []
        self.disagreement_trajectories = []
        self.a1_s_a_values = []
        self.a2_s_a_values = []
        self.a2_values_for_a1_states = []
        self.a1_values_for_a2_states = []

    def update(self, state_object, obs, a, a1_s_a_values, a2_values_for_a1_states, r, done,
               infos, position):
        self.obs.append(obs)
        self.positions.append(position)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states.append(state_object)
        self.length += 1
        self.a1_s_a_values.append(a1_s_a_values)
        self.a2_values_for_a1_states.append(a2_values_for_a1_states)
        self.a1_max_q_val = max(max(a1_s_a_values), self.a1_max_q_val)
        self.a2_max_q_val = max(max(a2_values_for_a1_states), self.a2_max_q_val)
        self.a1_min_q_val = min(min(a1_s_a_values), self.a1_min_q_val)
        self.a2_min_q_val = min(min(a2_values_for_a1_states), self.a2_min_q_val)

    def get_trajectories(self):
        """for each trajectory of agent 2 - find corresponding trajectory of agent 1"""
        for i, a2_traj in enumerate(self.a2_trajectories):
            start_idx, end_idx = a2_traj[0].id[1], a2_traj[-1].id[1]
            a1_traj = self.states[start_idx:end_idx + 1]
            a1_traj_q_values = [x.action_values for x in a1_traj]
            a2_traj_q_values = [x.action_values for x in a2_traj]
            a1_traj_indexes = [x.id[1] for x in a1_traj]
            a2_traj_indexes = list(range(start_idx, end_idx + 1))
            dt = DisagreementTrajectory(self.disagreement_indexes[i], a1_traj_indexes,
                                        a2_traj_indexes, self.trajectory_length, self.episode, i,
                                        a1_traj_q_values, a2_traj_q_values,
                                        self.a1_values_for_a2_states[i],
                                        self.a2_values_for_a1_states[start_idx:end_idx + 1],
                                        self.agent_ratio)
            self.a1_trajectory_indexes.append(a1_traj_indexes)
            self.disagreement_trajectories.append(dt)

    def get_frames(self, s1_indexes, s2_indexes, s2_traj):
        a1_frames = [self.states[x].image for x in s1_indexes]
        a2_frames = [self.a2_trajectories[s2_traj][x - min(s2_indexes)].image for x in s2_indexes]
        assert len(a1_frames) == self.trajectory_length, 'Error in highlight frame length'
        assert len(a2_frames) == self.trajectory_length, 'Error in highlight frame length'
        # plt.imshow(self.a2_trajectories[s2_traj][0].image)
        # plt.show()
        return a1_frames, a2_frames


class State(object):
    def __init__(self, idx, episode, obs, state, action_values, img, position, **kwargs):
        self.observation = obs
        self.image = img
        self.state = state
        self.position = position
        self.action_values = action_values
        self.id = (episode, idx)
        self.kwargs = kwargs

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


class DisagreementTrajectory(object):
    def __init__(self, da_index, a1_states, a2_states, horizon, episode, i, a1_s_a_values,
                 a2_s_a_values, a1_values_for_a2_states, a2_values_for_a1_states, agent_ratio):
        self.a1_states = a1_states
        self.a2_states = a2_states
        self.episode = episode
        self.trajectory_index = i
        self.horizon = horizon
        self.da_index = da_index
        self.disagreement_score = None
        self.importance = None
        self.state_importance_list = []
        self.agent_ratio = agent_ratio
        self.a1_s_a_values = a1_s_a_values
        self.a2_s_a_values = a2_s_a_values
        self.a1_values_for_a2_states = a1_values_for_a2_states
        self.a2_values_for_a1_states = a2_values_for_a1_states
        self.importance_funcs = {
            "max_min": trajectory_importance_max_min,
            "max_avg": trajectory_importance_max_avg,
            "avg": trajectory_importance_avg,
            "avg_delta": trajectory_importance_avg_delta,
        }

    def calculate_state_disagreement_extent(self, importance):
        self.state_importance = importance
        da_idx = self.da_index
        traj_da_idx = self.a1_states.index(da_idx)
        s1_vals, s2_vals = self.a1_s_a_values[traj_da_idx], self.a2_s_a_values[traj_da_idx]
        if importance == 'sb':
            return self.second_best_confidence(s1_vals, s2_vals)
        elif importance == 'bety':
            return self.better_than_you_confidence(s1_vals, s2_vals)

    def calculate_trajectory_importance(self, trace, i, importance):
        """calculate trajectory score"""
        s_i, e_i = self.a1_states[0], self.a1_states[-1]
        self.trajectory_importance = importance
        rel_idx = e_i - s_i
        if importance == "last_state":
            s1, s2 = trace.states[e_i], trace.a2_trajectories[i][rel_idx]
            return self.trajectory_importance_last_state(s1, s2, rel_idx)
        else:
            return self.get_trajectory_importance(importance, rel_idx)

    def get_trajectory_importance(self, importance, end):
        """state values"""
        s1_a1_vals = self.a1_s_a_values
        s1_a2_vals = self.a2_values_for_a1_states
        s2_a1_vals = self.a1_values_for_a2_states[:end + 1]
        s2_a2_vals = self.a2_s_a_values[:end + 1]
        """calculate value of all individual states in both trajectories,
         as ranked by both agents"""
        traj1_states_importance, traj2_states_importance = [], []
        for i in range(len(s1_a1_vals)):
            traj1_states_importance.append(self.get_state_value(s1_a1_vals[i], s1_a2_vals[i]))
            traj2_states_importance.append(self.get_state_value(s2_a1_vals[i], s2_a2_vals[i]))
        """calculate score of trajectories"""
        traj1_score = self.importance_funcs[importance](traj1_states_importance)
        traj2_score = self.importance_funcs[importance](traj2_states_importance)
        """return the difference between them. bigger == greater disagreement"""
        return abs(traj1_score - traj2_score)

    def trajectory_importance_last_state(self, s1, s2, idx):
        if s1.image.tolist() == s2.image.tolist(): return 0
        """state values"""
        s1_a1_vals = self.a1_s_a_values[-1]
        s1_a2_vals = self.a2_values_for_a1_states[-1]
        s2_a1_vals = self.a1_values_for_a2_states[idx]
        s2_a2_vals = self.a2_s_a_values[idx]
        """the value of the state is defined by the best available action from it"""
        s1_score = max(s1_a1_vals) * self.agent_ratio + max(s1_a2_vals)
        s2_score = max(s2_a1_vals) * self.agent_ratio + max(s2_a2_vals)
        return abs(s1_score - s2_score)

    def second_best_confidence(self, a1_vals, a2_vals):
        """compare best action to second-best action"""
        sorted_1 = sorted(a1_vals, reverse=True)
        sorted_2 = sorted(a2_vals, reverse=True)
        a1_diff = sorted_1[0] - sorted_1[1] * self.agent_ratio
        a2_diff = sorted_2[0] - sorted_2[1]
        return a1_diff + a2_diff

    def better_than_you_confidence(self, a1_vals, a2_vals):
        a1_diff = (max(a1_vals) - a1_vals[np.argmax(a2_vals)]) * self.agent_ratio
        a2_diff = max(a2_vals) - a2_vals[np.argmax(a1_vals)]
        return a1_diff + a2_diff

    def get_state_value(self, a1_vals, a2_vals):
        """
        the value of the state is defined by the best available action from it, as this is
        calculated by estimated future returns
        """
        return max(a1_vals) * self.agent_ratio + max(a2_vals)

    def normalize_q_values(self, a1_max, a1_min, a2_max, a2_min):
        self.a1_s_a_values = (np.array(self.a1_s_a_values) - a1_min) / (a1_max - a1_min)
        self.a2_s_a_values = (np.array(self.a2_s_a_values) - a2_min) / (a2_max - a2_min)
        self.a1_values_for_a2_states = \
            (np.array(self.a1_values_for_a2_states) - a1_min) / (a1_max - a1_min)
        self.a2_values_for_a1_states = \
            (np.array(self.a2_values_for_a1_states) - a2_min) / (a2_max - a2_min)


def disagreement(timestep, trace, env2, a2, curr_s, a1):
    trajectory_states, trajectory_scores = \
        disagreement_states(trace, env2, a2, timestep, curr_s)
    a2_s_a_values = [x.action_values for x in trajectory_states]
    a1_values_for_a2_states = [a1.get_state_action_values(x.state) for x in trajectory_states]
    trace.a2_s_a_values.append(a2_s_a_values)
    trace.a2_trajectories.append(trajectory_states)
    trace.a2_rewards.append(trajectory_scores)
    trace.disagreement_indexes.append(timestep)
    trace.a1_values_for_a2_states.append(a1_values_for_a2_states)


def save_disagreements(a1_DAs, a2_DAs, output_dir, fps):
    disagreement_frames_dir = join(output_dir, "disagreement_frames")
    video_dir = join(output_dir, "videos")
    make_clean_dirs(video_dir)
    make_clean_dirs(join(video_dir, 'temp'))
    make_clean_dirs(disagreement_frames_dir)
    dir = join(video_dir, 'temp')

    height, width, layers = a1_DAs[0][0].shape
    size = (width, height)
    trajectory_length = len(a1_DAs[0])
    da_idx = trajectory_length // 2
    for da_i in range(len(a1_DAs)):
        for img_i in range(len(a1_DAs[da_i])):
            save_image(disagreement_frames_dir, "a1_DA{}_Frame{}".format(str(da_i), str(img_i)),
                       a1_DAs[da_i][img_i])
            save_image(disagreement_frames_dir, "a2_DA{}_Frame{}".format(str(da_i), str(img_i)),
                       a2_DAs[da_i][img_i])

        """up to disagreement"""
        create_video('together' + str(da_i), disagreement_frames_dir, dir, "a1_DA" + str(da_i), size,
                     da_idx, fps, add_pause=[0, 4])
        """from disagreement"""
        name1, name2 = "a1_DA" + str(da_i), "a2_DA" + str(da_i)
        create_video(name1, disagreement_frames_dir, dir, name1, size,
                     trajectory_length, fps, start=da_idx)
        # create_video(name2, disagreement_frames_dir, dir, name2, size,
        #              trajectory_length, fps, start=da_idx)
    return video_dir


def get_pre_disagreement_states(t, horizon, states):
    start = t - (horizon // 2) + 1
    pre_disagreement_states = []
    if start < 0:
        pre_disagreement_states = [states[0] for _ in range(abs(start))]
        start = 0
    pre_disagreement_states = pre_disagreement_states + states[start:]
    return pre_disagreement_states


def disagreement_states(trace, env, agent, timestep, curr_s):
    horizon, da_rewards = env.args.horizon, []
    start = timestep - (horizon // 2) + 1
    if start < 0: start = 0
    trajectory_states = trace.states[start:]
    da_state = deepcopy(trajectory_states[-1])
    da_state.action_values = agent.get_state_action_values(curr_s)
    trajectory_states[-1] = da_state
    done = False
    next_timestep = timestep + 1
    for step in range(next_timestep, next_timestep + (horizon // 2)):
        if done: break
        a = agent.act(curr_s)
        new_obs, r, done, info = env.step(a)
        new_s = new_obs
        new_s_a_values = agent.get_state_action_values(new_s)
        new_frame = env.render(mode='rgb_array')
        position = deepcopy(env.road.vehicles[0])
        new_state = State(step, trace.episode, new_obs, new_s, new_s_a_values, new_frame, position)
        trajectory_states.append(new_state)
        da_rewards.append(r)
        curr_s = new_s
        trace.a2_max_q_val = max(max(new_s_a_values), trace.a2_max_q_val)
        trace.a2_min_q_val = min(min(new_s_a_values), trace.a2_min_q_val)
    return trajectory_states, da_rewards


def get_top_k_disagreements(traces, args):
    """obtain the N-most important trajectories"""
    top_k_diverse_trajectories, discarded_context = [], []
    """get all trajectories"""
    all_trajectories = []
    for trace in traces:
        all_trajectories += [t for t in trace.disagreement_trajectories]
    sorted_trajectories = sorted(all_trajectories, key=lambda x: x.importance, reverse=True)
    """select trajectories"""
    seen_indexes = {i: [] for i in range(len(traces))}
    for d in sorted_trajectories:
        t_indexes = d.a1_states
        intersecting_indexes = set(seen_indexes[d.episode]).intersection(set(t_indexes))
        if len(intersecting_indexes) > args.similarity_limit:
            discarded_context.append(d)
            continue
        seen_indexes[d.episode] += t_indexes
        top_k_diverse_trajectories.append(d)
        if len(top_k_diverse_trajectories) == args.n_disagreements:
            break

    if not len(top_k_diverse_trajectories) == args.n_disagreements:
        top_k_diverse_trajectories += discarded_context
    top_k_diverse_trajectories = top_k_diverse_trajectories[:args.n_disagreements]

    log(args.logger, f'Chosen disagreements:', args.verbose)
    for d in top_k_diverse_trajectories:
        log(args.logger, f'Name: ({d.episode},{d.da_index})')

    return top_k_diverse_trajectories


def make_same_length(trajectories, horizon, traces):
    """make all trajectories the same length"""
    for d in trajectories:
        if len(d.a1_states) < horizon:
            """insert to start of video"""
            da_traj_idx = d.a1_states.index(d.da_index)
            for _ in range((horizon // 2) - da_traj_idx - 1):
                d.a1_states.insert(0, d.a1_states[0])
                d.a2_states.insert(0, d.a1_states[0])
            """insert to end of video"""
            while len(d.a1_states) < horizon:
                last_idx = d.a1_states[-1]
                if last_idx < len(traces[d.episode].states) - 1:
                    last_idx += 1
                    d.a1_states.append(last_idx)
                else:
                    d.a1_states.append(last_idx)

        for _ in range(horizon - len(d.a2_states)):
            d.a2_states.append(d.a2_states[-1])
    return trajectories
