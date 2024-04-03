import torch as T

from Car import Car
from .Skin import Skin


class RaceCar(Car):
    def __init__(self, dtype: T.dtype, device: str):
        car_length = 4.0
        car_width = 2.0
        car_height = 2.0

        car_wheelbase_ratio = 0.9
        car_track_ratio = 1.0

        max_speed = 50
        max_wheel_angle = T.pi * 24 / 180
        max_acceleration = 40.0

        fov = 3 * T.pi / 2
        n_rays = 11
        ray_range = 50.0

        car_color = (0, 200, 255)
        crashed_car_color = (255, 0, 0)
        vision_lines_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        car_skin = Skin(car_color, crashed_car_color, vision_lines_color, outline_color)

        super(RaceCar, self).__init__(
            car_length,
            car_width,
            car_height,
            car_wheelbase_ratio,
            car_track_ratio,
            max_speed,
            max_wheel_angle,
            max_acceleration,
            fov,
            n_rays,
            ray_range,
            car_skin,
            dtype=dtype,
            device=device,
        )
