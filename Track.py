import os
import torch as T

class Track:
    def __init__(self, left_rails : T.Tensor, right_rails : T.Tensor, points : T.Tensor):
        self.left_rails = left_rails
        self.right_rails = right_rails
        self.points = points

    @classmethod
    def Load(self, track_number : int) -> "Track":
        try:
            track_data = T.load(f"Tracks//Track-{track_number}//data.pt")
            return Track(track_data["left_rails"], track_data["right_rails"], track_data["points"])
        
        except FileNotFoundError:
            track_names = os.listdir("Tracks")
            raise FileNotFoundError(f"Track number '{track_number}' not valid. Choose between {'-' if len(track_names) == 0 else 1}-{'-' if len(track_names) == 0 else len(track_names)}")
