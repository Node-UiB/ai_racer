import pygame
import pygame.gfxdraw as gfxdraw
import pickle
import numpy as np


class TrackData:
    def __init__(self, start_position: tuple[int, int], track_data: np.ndarray):
        self.start_position = start_position
        self.track_data = track_data


class Track:
    def __init__(self, surface: pygame.Surface, loop=True):
        self.surface = surface
        self.surface_size = surface.get_size()
        self.loop = loop
        self.checkpoints = []
        self.outer_lines = []
        self.inner_lines = []
        self.start_position = None
        self.track_color = (128, 128, 128)
        self.finish_color = (210, 128, 128, 200)

    def draw(self):
        if self.checkpoints:
            for checkpoint, outer, inner in zip(
                self.checkpoints, self.outer_lines, self.inner_lines
            ):
                # pygame.draw.circle(surface, "green", checkpoint, 5)
                # pygame.draw.circle(self.surface, "grey", outer, 5)
                # pygame.draw.circle(self.surface, "grey", inner, 5)
                pass

            if len(self.checkpoints) >= 2:
                pygame.draw.lines(self.surface, "grey", self.loop, self.outer_lines)
                pygame.draw.lines(self.surface, "grey", self.loop, self.inner_lines)

    def draw_poly(self):
        for i in range(len(self.checkpoints)):
            if i == len(self.checkpoints) - 1:
                if self.loop:
                    gfxdraw.filled_polygon(
                        self.surface,
                        (
                            self.outer_lines[0],
                            self.outer_lines[-1],
                            self.inner_lines[-1],
                            self.inner_lines[0],
                        ),
                        self.finish_color,
                    )
            elif i != -1:
                if i == 0:
                    color = (128, 210, 128)
                elif not self.loop and i == len(self.checkpoints) - 2:
                    color = self.finish_color
                else:
                    color = self.track_color
                gfxdraw.filled_polygon(
                    self.surface,
                    (
                        self.outer_lines[i],
                        self.outer_lines[i + 1],
                        self.inner_lines[i + 1],
                        self.inner_lines[i],
                    ),
                    color,
                )

    def add_point(self, point: tuple[int, int]):
        if self.is_inside_surface(point):
            self.checkpoints.append(point)
            self.outer_lines.append(None)
            self.inner_lines.append(None)

    def remove_point(self):
        if self.checkpoints:
            self.checkpoints.pop()
            self.outer_lines.pop()
            self.inner_lines.pop()

    def get_side_points(self):
        pos = pygame.mouse.get_pos()
        if self.is_inside_surface(pos):
            point = self.checkpoints[-1]
            dx, dy = pos[0] - point[0], pos[1] - point[1]
            opp_pos = point[0] - dx, point[1] - dy

            self.outer_lines[-1] = opp_pos
            self.inner_lines[-1] = pos

    def is_inside_surface(self, point: tuple[int, int]) -> bool:
        return point[0] < self.surface_size[0] and point[1] < self.surface_size[1]

    def save(self):
        imgdata = pygame.surfarray.array3d(self.surface)
        imgdata = imgdata.swapaxes(0, 1)
        track_data = TrackData(self.checkpoints[0], imgdata)
        with open("tracks/new_track.pickl", "wb") as f:
            pickle.dump(track_data, f)
