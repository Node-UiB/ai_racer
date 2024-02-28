import os
import torch as T

from linalg import get_lines

from typing import Tuple


class Track:
    def __init__(self, 
                 left_rails: T.Tensor, 
                 right_rails: T.Tensor, 
                 points: T.Tensor, 
                 track_color : Tuple[int, int, int], 
                 rail_color : Tuple[int, int, int], 
                 goal_line_color : Tuple[int, int, int],
                 track_name : str = "Track"):
        
        self.left_rails = left_rails
        self.right_rails = right_rails
        self.points = points

        self.track_color = track_color
        self.rail_color = rail_color
        self.goal_line_color = goal_line_color

        self.track_name = track_name

        self.SetTrackLines()


    def SetTrackLines(self):
        left_rail_lines = get_lines(self.left_rails, closed=True)
        right_rail_lines = get_lines(self.right_rails, closed=True)

        self.track_lines = T.concat((left_rail_lines, right_rail_lines), dim=0)


    @classmethod
    def Load(self, track_name: str,
             dtype : T.dtype, 
             device : str = "cpu", 
             track_color : Tuple[int, int, int] = (35, 30, 30), 
             rail_color : Tuple[int, int, int] = (0, 0, 0),
             goal_line_color : Tuple[int, int, int] = (255, 255, 255)) -> "Track":
        
        try:
            track_data = T.load(f"Tracks//{track_name}//data.pt")

            return Track(
                track_data["left_rails"].to(dtype=dtype, device=device),
                track_data["right_rails"].to(dtype=dtype, device=device),
                track_data["points"].to(dtype=dtype, device=device),
                track_color=track_color,
                rail_color=rail_color,
                track_name=track_name,
                goal_line_color=goal_line_color
            )

        except FileNotFoundError:
            track_names = os.listdir("Tracks")
            raise FileNotFoundError(
                f"Track: '{track_name}' not valid. Choose between: {', '.join(track_names)}"
            )
