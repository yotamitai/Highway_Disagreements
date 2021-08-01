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
        cars_x_dist = [car[1] for car in other_cars if abs(car[2]) > 0.1]
        closest_car = 1 - lmap(min(cars_x_dist), [0, 0.4], [0, 1]) if cars_x_dist \
            else 0
        reward = \
            + self.config["distance_reward"] * closest_car \
            + self.config["collision_reward"] * self.vehicle.crashed
        reward = lmap(reward, [self.config["collision_reward"], self.config["distance_reward"]],
                      [0, 1])
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
        dist = 0
        max_dist = math.sqrt(0.4 ** 2 + 0.75 ** 2) # max in x and y coords relative to agent
        for i, car in enumerate(other_cars):
            dist += math.sqrt(abs(car[1]) ** 2 + abs(car[2]) ** 2)
        scaled_dist = lmap(dist, [0, 4 * max_dist], [0, 1])
        reward = \
            + self.config['distance_reward'] * scaled_dist \
            + self.config["collision_reward"] * self.vehicle.crashed
        reward = lmap(reward, [self.config["collision_reward"], self.config['distance_reward']],
                      [0, 1])
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
            else 1
        reward = \
            + self.config["high_speed_reward"] * scaled_speed \
            + self.config["distance_reward"] * closest_car \
            + self.config["collision_reward"] * self.vehicle.crashed
        reward = lmap(reward,
                      [self.config["collision_reward"],
                       self.config["distance_reward"] + self.config["high_speed_reward"]],
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
    id='FastRight-v0',
    entry_point='highway_disagreements.configs.reward_functions:FastRight',
)
