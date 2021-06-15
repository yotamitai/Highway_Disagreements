from os.path import join

import xxhash

from utils import Trace, State, pickle_save, pickle_load


def get_traces(environment, agent, agent_args, args):
    """
    kwargs: args of agent
    args: args of highlights algorithm
    """
    if args.load_traces:
        """Load traces and state dictionary"""
        execution_traces = pickle_load(join(args.results_dir,'Traces.pkl'))
        states_dictionary = pickle_load(join(args.results_dir,'States.pkl'))
        if args.verbose: print(f"Highlights {15*'-'+'>'} Traces & States Loaded")
    else:
        """Obtain traces and state dictionary"""
        execution_traces, states_dictionary = [], {}
        for i in range(args.n_traces):
            get_single_trace(environment, agent, i, execution_traces, states_dictionary, agent_args, args)
            if args.verbose: print(f"\tTrace {i} {15*'-'+'>'} Obtained")
        """save to results dir"""
        pickle_save(execution_traces, join(args.results_dir, 'Traces.pkl'))
        pickle_save(states_dictionary, join(args.results_dir, 'States.pkl'))
        if args.verbose: print(f"Highlights {15*'-'+'>'} Traces & States Generated")

    return execution_traces, states_dictionary


def get_single_trace(env, agent, trace_idx, agent_traces, states_dict, kwargs, args):
    """Implement a single trace while using the Trace and State classes"""
    trace = Trace()
    # ********* Implement here *****************
    curr_obs = env.reset()
    done = False
    while not done:
        a = agent.act(curr_obs)
        obs, r, done, infos = env.step(a)


        # *******************************
        """Add step to trace"""
        trace.reward_sum += r
        trace.length += 1
        trace.obs.append(obs), trace.rewards.append(r)
        trace.dones.append(done), trace.infos.append(infos)
        trace.actions.append(a)
        """state"""
        state_img = env.render()
        state_q_values = None
        features = None
        state_id = xxhash.xxh64(state_img.tobytes(), seed=0).hexdigest()
        if state_id not in states_dict.keys():
            states_dict[state_id] = State(state_id, obs, state_q_values, features, state_img)
        trace.states.append(state_id)

    agent_traces.append(trace)
