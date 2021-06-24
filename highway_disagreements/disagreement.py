import logging
from copy import copy
from os.path import join
import gym
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from get_trajectories import trajectory_importance_max_min, \
    trajectory_importance_max_avg, trajectory_importance_avg, trajectory_importance_avg_delta
from highway_disagreements.utils import save_image, create_video, make_clean_dirs


class DisagreementTrace(object):
    def __init__(self, episode, trajectory_length, agent_ratio=1):
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
        self.a1_rewards = []
        self.a2_rewards = []
        self.agent_ratio = agent_ratio
        self.disagreement_indexes = []
        self.importance_scores = []
        self.disagreement_trajectories = []
        self.diverse_trajectories = []
        self.a1_s_a_values = []
        self.a2_s_a_values = []
        self.a2_values_for_a1_states = []
        self.a1_values_for_a2_states = []

    def update(self, state_object, obs, a, a1_s_a_values, a2_values_for_a1_states, r, done,
               infos):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states.append(state_object)
        self.length += 1
        self.a1_s_a_values.append(a1_s_a_values)
        self.a2_values_for_a1_states.append(a2_values_for_a1_states)

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
                                        a2_traj_indexes, self.trajectory_length, self.episode,
                                        a1_traj_q_values, a2_traj_q_values,
                                        self.a1_values_for_a2_states[i],
                                        self.a2_values_for_a1_states[start_idx:end_idx + 1],
                                        self.agent_ratio)
            self.a1_trajectory_indexes.append(a1_traj_indexes)
            self.disagreement_trajectories.append(dt)


class State(object):
    def __init__(self, idx, episode, obs, state, action_values, img, position, **kwargs):
        self.observation = obs
        self.image = img
        self.state = state
        self.action_values = action_values
        self.id = (episode, idx)
        self.position = position
        self.kwargs = kwargs

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


class DisagreementTrajectory(object):
    def __init__(self, da_index, a1_states, a2_states, horizon, episode, a1_s_a_values,
                 a2_s_a_values, a1_values_for_a2_states, a2_values_for_a1_states, agent_ratio):
        self.a1_states = a1_states
        self.a2_states = a2_states
        self.episode = episode
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
            "max_avg": trajectory_importance_max_min,
            "avg": trajectory_importance_max_min,
            "avg_delta": trajectory_importance_max_min,
        }

    def calculate_state_importance(self, s1, s2, importance):
        return self.calculate_disagreement_score(s1, s2, importance)

    def calculate_trajectory_importance(self, trajectory_importance, state_importance, a1_states,
                                        a2_states):
        """calculate trajectory score"""
        self.trajectory_importance = trajectory_importance
        self.state_importance = state_importance
        if trajectory_importance == "last_state":
            return self.trajectory_importance_last_state(a1_states[-1], a2_states[-1])
        else:
            return self.get_trajectory_importance(a1_states, a2_states, trajectory_importance,
                                                  state_importance)

    def get_trajectory_importance(self, s1, s2, trajectory_importance, state_importance):
        """state values"""
        s1_a1_vals = self.a1_s_a_values
        s1_a2_vals = self.a2_values_for_a1_states
        s2_a1_vals = self.a1_values_for_a2_states
        s2_a2_vals = self.a2_s_a_values
        """calculate value of all individual states in both trajectories,
         as ranked by both agents"""
        traj1_importance_of_states = [
            self.calculate_state_importance(s1_a1_vals[i], s1_a2_vals[i], state_importance) for i
            in range(len(s1_a1_vals))]
        traj2_importance_of_states = [
            self.calculate_state_importance(s2_a1_vals[i], s2_a2_vals[i], state_importance) for i
            in range(len(s2_a2_vals))]
        """calculate score of trajectories"""
        traj1_score = self.importance_funcs[trajectory_importance](traj1_importance_of_states)
        traj2_score = self.importance_funcs[trajectory_importance](traj2_importance_of_states)
        """return the difference between them. bigger == greater disagreement"""
        return abs(traj1_score - traj2_score)

    def trajectory_importance_last_state(self, s1, s2):
        """state values"""
        if s1.state.tolist() == s2.state.tolist(): return 0
        s1_a1_vals = self.a1_s_a_values[-1]
        s1_a2_vals = self.a2_values_for_a1_states[-1]
        s2_a1_vals = self.a1_values_for_a2_states[-1]
        s2_a2_vals = self.a2_s_a_values[-1]
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

    def calculate_disagreement_score(self, s1_vals, s2_vals, importance):
        a1_vals = softmax(s1_vals)
        a2_vals = softmax(s2_vals)
        if importance == 'sb':
            return self.second_best_confidence(a1_vals, a2_vals)
        elif importance == 'bety':
            return self.better_than_you_confidence(a1_vals, a2_vals)

    def get_frames(self):
        a1_frames = [x.image for x in self.a1_states]
        a2_frames = [x.image for x in self.a2_states]
        if len(a1_frames) != self.horizon:
            a1_frames = a1_frames + [a1_frames[-1] for _ in range(self.horizon - len(a1_frames))]
            a2_frames = a2_frames + [a2_frames[-1] for _ in range(self.horizon - len(a2_frames))]
        return a1_frames, a2_frames


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
    highlight_frames_dir = join(output_dir, "highlight_frames")
    video_dir = join(output_dir, "videos")
    make_clean_dirs(video_dir)
    make_clean_dirs(highlight_frames_dir)

    height, width, layers = a1_DAs[0][0].shape
    size = (width, height)
    trajectory_length = len(a1_DAs[0])
    for hl_i in range(len(a1_DAs)):
        for img_i in range(len(a1_DAs[hl_i])):
            save_image(highlight_frames_dir, "a1_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a1_DAs[hl_i][img_i])
            save_image(highlight_frames_dir, "a2_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a2_DAs[hl_i][img_i])

        create_video(highlight_frames_dir, video_dir, "a1_DA" + str(hl_i), size,
                     trajectory_length, fps)
        create_video(highlight_frames_dir, video_dir, "a2_DA" + str(hl_i), size,
                     trajectory_length, fps)
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
    # obtain last pre-disagreement states
    # da_states = get_pre_disagreement_states(timestep, horizon, trace.states)
    start = timestep - (horizon // 2) + 1
    if start < 0: start = 0
    da_states = trace.states[start:]
    done = False
    for step in range(timestep + 1, timestep + (horizon // 2)):
        if done: break
        a = agent.act(curr_s)
        new_obs, r, done, info = env.step(a)
        new_s = new_obs
        new_s_a_values = agent.get_state_action_values(new_s)
        new_frame = env.render(mode='rgb_array')
        new_position = env.vehicle.position
        new_state = State(step, trace.episode, new_obs, new_s, new_s_a_values, new_frame,
                          new_position)
        da_states.append(new_state)
        da_rewards.append(r)
        curr_s = new_s

    # if len(da_states) < horizon:
    #     da_states += [da_states[-1] for _ in range(horizon - len(da_states))]
    #     da_rewards += [0 for _ in range(horizon - len(da_states))]
    return da_states, da_rewards


def get_top_k_disagreements(traces, args):
    """"""
    top_k_diverse_trajectories, discarded_context, discarded_importance = [], [], []
    """get diverse trajectories"""
    all_trajectories = []
    for trace in traces:
        all_trajectories += [t for t in trace.disagreement_trajectories]
    sorted_trajectories = sorted(all_trajectories, key=lambda x: x.importance, reverse=True)

    seen_importance = []
    seen_indexes = {i: [] for i in range(len(traces))}
    for t in sorted_trajectories:
        """last_state_loc: specific distances"""
        if args.trajectory_importance == "last_state_loc":
            if t.importance == 429: continue
        """last_state_val"""
        if args.trajectory_importance == "last_state_val":
            if t.states[-2].state == t.a2_states[-2].state:
                continue
            if t.a1_states[-1].state == 1036 or t.a2_states[-1].state == 1036:
                continue

        t_indexes = [s.id for s in t.a1_states]
        intersecting_indexes = set(seen_indexes[t.episode]).intersection(set(t_indexes))
        if len(intersecting_indexes) > args.similarity_limit:
            discarded_context.append(t)
            continue
        seen_indexes[t.episode] += t_indexes
        top_k_diverse_trajectories.append(t)
        if len(top_k_diverse_trajectories) == args.n_disagreements:
            break

    if not len(top_k_diverse_trajectories) == args.n_disagreements:
        top_k_diverse_trajectories += discarded_context
    if not len(top_k_diverse_trajectories) == args.n_disagreements:
        top_k_diverse_trajectories += discarded_importance
    for i in range(args.n_disagreements - len(top_k_diverse_trajectories)):
        top_k_diverse_trajectories.append(discarded_context[i])
    top_k_diverse_trajectories = top_k_diverse_trajectories[:args.n_disagreements]
    logging.info(f'Chosen disagreements:')
    for d in top_k_diverse_trajectories:
        logging.info(f'Name: {d.a1_states[(args.horizon // 2) - 1].id}, '
                     f'State: {d.a1_states[(args.horizon // 2) - 1].state} '
                     f'Position: {d.a1_states[(args.horizon // 2) - 1].agent_position}')

    """make all trajectories the same length"""
    for t in top_k_diverse_trajectories:
        if len(t.a1_states) < args.horizon:
            for _ in range(args.horizon - len(t.a1_states)):
                t.a1_states += [t.a1_states[-1]]
                t.a2_states += [t.a2_states[-1]]
    return top_k_diverse_trajectories
