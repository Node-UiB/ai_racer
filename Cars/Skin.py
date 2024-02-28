from typing import Tuple


class Skin:
    def __init__(
        self,
        car_color: Tuple[int, int, int],
        crashed_car_color: Tuple[int, int, int],
        vision_lines_color: Tuple[int, int, int],
        outline_color: Tuple[int, int, int],
    ):

        self.car_color = car_color
        self.crashed_car_color = crashed_car_color
        self.vision_lines_color = vision_lines_color
        self.outline_color = outline_color

