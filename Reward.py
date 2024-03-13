import torch as T

from Track import Track
from LinAlg import LinAlg

class Reward:
    def __init__(
            self,
            distance_reward_density : float,
            time_penalty : float,
            crash_penalty : float
    ):
        self.distance_reward_density = distance_reward_density
        self.time_penalty = time_penalty
        self.crash_penalty = crash_penalty

        self.last_distance = 0

    def __call__(self, car_position : T.Tensor, track : Track, is_crashed : bool, dt : float) -> T.Tensor:
        nearest_line_index, distance_along_line = LinAlg.get_distance_along_lines(car_position, track.way_point_lines)
        current_distance = T.sum(track.way_point_distances[:nearest_line_index], dim=0) + distance_along_line
        distance_delta = current_distance - self.last_distance
        self.last_distance = current_distance

        if is_crashed:
            self.last_distance = 0

        return self.distance_reward_density * distance_delta - is_crashed * self.crash_penalty - self.time_penalty * dt