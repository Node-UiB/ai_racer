import pickle
from PIL import Image


with open("tracks/new_track.pickl", "rb") as f:
    track = pickle.load(f)

print(track.start_position)
img = Image.fromarray(track.track_data, "RGB")
img.show()
