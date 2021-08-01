import math

import numpy as np

from highway_env.envs import HighwayEnv, Action
from gym.envs.registration import register

from highway_env.utils import lmap
from highway_env.vehicle.controller import ControlledVehicle


class HighwayEnvMinSpeed(HighwayEnv):

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.controlled_vehicles[0].SPEED_MIN = 10


class ParallelDriver(HighwayEnvMinSpeed):
    """rewarded for driving in parallel to a car"""

    def _reward(self, action: Action) -> float:
        obs = self.observation_type.observe()
        other_cars = obs[1:]
        # closest car that is not in same lane
        target_car_x_dist = [car[1] for car in other_cars if abs(car[2]) > 0.1][0]
        reward = 1 - target_car_x_dist \
                 + self.config["collision_reward"] * self.vehicle.crashed
        reward = lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


register(
    id='ParallelDriver-v0',
    entry_point='highway_disagreements.configs.reward_functions:ParallelDriver',
)


class SocialDistance(HighwayEnvMinSpeed):
    """rewarded for keeping as much distance from all cars"""

    def _reward(self, action: Action) -> float:
        other_cars = self.observation_type.observe()[1:]
        relativity = list(range(len(other_cars), 0, -1))
        x_reward = y_reward = 0
        for i, car in enumerate(other_cars):
            x_reward += abs(car[1]) * relativity[i]
            y_reward += abs(car[2]) * relativity[i]
        reward = lmap(y_reward, [0, 7.5], [0, 1]) + lmap(x_reward, [0, 4], [0, 1]) \
                 + self.config["collision_reward"] * self.vehicle.crashed
        reward = lmap(reward, [self.config["collision_reward"], 2], [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


register(
    id='SocialDistance-v0',
    entry_point='highway_disagreements.configs.reward_functions:SocialDistance',
)


class NoLaneChange(HighwayEnvMinSpeed):
    """penalized for changing lanes, otherwise rewarded for speed"""

    def _reward(self, action: Action) -> float:
        lane_change = action == 0 or action == 2
        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["high_speed_reward"] * scaled_speed \
            + self.config["lane_change_reward"] * lane_change
        reward = lmap(reward,
                      [self.config["collision_reward"] + self.config["lane_change_reward"],
                       self.config["high_speed_reward"]], [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


register(
    id='NoLaneChange-v0',
    entry_point='highway_disagreements.configs.reward_functions:NoLaneChange',
)


class ClearLane(HighwayEnv):

    def _reward(self, action: Action) -> float:
        """ if no cars in your lane - max reward,
         else reward based on how close agent is to a car in it's lane"""
        obs = self.observation_type.observe()
        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        other_cars = obs[1:]
        dist_cars_in_front = [x[1] for x in other_cars if x[1] > 0 and abs(x[2]) <= 0.05]
        closest_car = lmap(min(dist_cars_in_front), [0, 0.4], [0, 1]) if dist_cars_in_front \
            else 0
        reward = \
            + self.config["reward_speed_range"] * scaled_speed \
            + self.config["keep_distance_reward"] * closest_car \
            + self.config["collision_reward"] * self.vehicle.crashed
        reward = lmap(reward,
                      [self.config["collision_reward"],
                       self.config["keep_distance_reward"] + self.config["reward_speed_range"]],
                      [0, 1])
        return reward


register(
    id='ClearLane-v0',
    entry_point='highway_disagreements.configs.reward_functions:ClearLane',
)


class FastRight(HighwayEnv):

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
        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = lmap(reward,
                      [self.config["collision_reward"],
                       self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                      [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


register(
    id='fastRight-v0',
    entry_point='highway_disagreements.configs.reward_functions:FastRight',
)
