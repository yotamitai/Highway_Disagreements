from numpy import argmax


def rank_trajectories(traces, importance_type, state_importance, traj_importance):
    for trace in traces:
        a1_q_max, a2_q_max = trace.a1_max_q_val, trace.a2_max_q_val
        a1_q_min, a2_q_min = trace.a1_min_q_val, trace.a2_min_q_val
        for i, d in enumerate(trace.disagreement_trajectories):
            d.normalize_q_values(a1_q_max, a1_q_min, a2_q_max, a2_q_min)
            if importance_type == 'state':
                importance = d.calculate_state_importance(state_importance)
            else:
                importance = d.calculate_trajectory_importance(trace, i, traj_importance,
                                                               state_importance)

            # this is a fix - need to understand why this happens sometimes
            relative_idx = d.da_index - d.a1_states[0]
            if argmax(d.a1_s_a_values[relative_idx]) == argmax(d.a2_s_a_values[relative_idx]):
                importance = 0
            d.importance = importance


def trajectory_importance_max_min(states_importance):
    """ computes the importance of the trajectory, according to max-min approach:
     delta(max state, min state) """
    return max(states_importance) - min(states_importance)


def trajectory_importance_max_avg(states_importance):
    """ computes the importance of the trajectory, according to max-avg approach:
     delta(max state, avg) """
    avg = sum(states_importance) / len(states_importance)
    return max(states_importance) - avg


def trajectory_importance_avg(states_importance):
    """ computes the importance of the trajectory, according to avg approach """
    avg = sum(states_importance) / len(states_importance)
    return avg


def trajectory_importance_avg_delta(states_importance):
    """ computes the importance of the trajectory, according to the average delta approach """
    sum_delta = 0
    for i in range(len(states_importance)):
        sum_delta += states_importance[i] - states_importance[i - 1]
    avg_delta = sum_delta / len(states_importance)
    return avg_delta
