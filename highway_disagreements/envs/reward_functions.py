import numpy as np
from gym.envs.registration import register
from numpy import where

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
        self.observation_type.observe()
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


class OnlySafe(LocalHighwayEnv):

    def _reward(self, action: Action) -> float:
        reward = 1 if not self.vehicle.crashed else self.config["collision_reward"]
        reward = 0 if not self.vehicle.on_road else reward
        return reward


class OnlySpeed(LocalHighwayEnv):

    def _reward(self, action: Action) -> float:
        reward = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


class ClearLane(LocalHighwayEnv):

    def _reward(self, action: Action) -> float:
        """ if no cars in your lane - max reward,
         else reward based on how close agent is to a car in it's lane"""
        obs = self.observation_type.observe()
        cars_in_lane_in_front = [x for x in obs if x[1] > 0 and abs(x[2]) <= 0.05]
        reward = 1 if not cars_in_lane_in_front else min(cars_in_lane_in_front[0][1], 1)
        if self.vehicle.crashed: reward = 0
        if not self.vehicle.on_road: reward = 0
        return reward


register(
    id='fastRight-v0',
    entry_point='highway_disagreements.envs.reward_functions:FastRight',
)
register(
    id='onlySpeed-v0',
    entry_point='highway_disagreements.envs.reward_functions:OnlySpeed',
)
register(
    id='onlySafe-v0',
    entry_point='highway_disagreements.envs.reward_functions:OnlySafe',
)

register(
    id='clearLane-v0',
    entry_point='highway_disagreements.envs.reward_functions:ClearLane',
)
