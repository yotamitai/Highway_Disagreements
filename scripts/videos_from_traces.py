import argparse
import json
import os
from os.path import abspath, join
from pathlib import Path

from highway_disagreements.compare_agents import main

if __name__ == '__main__':

    # directories = [
    #     '/home/yotama/OneDrive/Local_Git/Highway_Disagreements/User Study Videos/DA/DA_state_bety_6fps_20frames/2021-08-04_13:10:11_SocialDistance-ClearLane',
    #     '/home/yotama/OneDrive/Local_Git/Highway_Disagreements/User Study Videos/DA/DA_state_bety_6fps_20frames/2021-08-04_13:41:48_ClearLane-FastRight'
    # ]

    base_dir = '/home/yotama/OneDrive/Local_Git/Highway_Disagreements/User Study Videos/DA/DA_trajLastState_bety_7fps_20frames'
    directories = os.listdir(base_dir)

    """RUN"""
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')

    for d in directories:
        f = open(Path(join(base_dir, d, 'metadata.json')))
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        args.traces_path = join(base_dir, d)

        """get more/less trajectories"""
        # args.similarity_limit = 3  # int(args.horizon * 0.66)
        """importance measures"""
        args.state_importance = "bety"  # "sb" "bety"
        args.trajectory_importance = "last_state"  # last_state, max_min, max_avg, avg, avg_delta
        args.importance_type = 'trajectory'  # state/trajectory
        args.verbose = False
        args.fps = 7
        args.randomized = True

        main(args)
