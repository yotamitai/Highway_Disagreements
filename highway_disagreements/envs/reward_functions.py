import numpy as np
from gym.envs.registration import register

from highway_disagreements.envs.highway_local import LocalHighwayEnv
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle


class FastRight(LocalHighwayEnv):

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


register(
    id='fastRight-v0',
    entry_point='highway_disagreements.envs.reward_functions:FastRight',
)


class alwaysOne(LocalHighwayEnv):

    def _reward(self, action: Action) -> float:
        return 1

register(
    id='alwaysOne-v0',
    entry_point='highway_disagreements.envs.reward_functions:alwaysOne',
)