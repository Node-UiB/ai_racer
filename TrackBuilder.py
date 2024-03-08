import os
import sys
import torch as T
import numpy as np
import pygame as pg
from math import floor
from typing import List, Tuple


class TrackBuilder:
    def __init__(self):
        self.SetAttributes()
        self.SetScreen()
        self.SetValueBinds()
        self.ResetTrack()
        self.UpdateEvents()

        self.Run()

    def SetScreen(self):
        pg.init()
        display_info = pg.display.Info()
        self.screen_x, self.screen_y = display_info.current_w, display_info.current_h

        self.screen = pg.display.set_mode((self.screen_x, self.screen_y), pg.FULLSCREEN)
        pg.mouse.set_visible(False)

        self.screen_pos = (self.screen_x / 2.0, self.screen_y / 2.0)

    def SetValueBinds(self):
        self.key_value_binds = {}
        self.mouse_value_binds = {}

        for event_type, key_binds in self.key_binds.items():
            event_type_value = getattr(pg, f"{event_type}")
            self.key_value_binds[event_type_value] = {}

            for key, method in key_binds.items():
                try:
                    value = getattr(pg, f"K_{key}")
                    self.key_value_binds[event_type_value][value] = method

                except AttributeError:
                    print(f"Key '{key}' not an attribute of pygame.")
                    continue

        for event_type, mouse_binds in self.mouse_binds.items():
            event_type_value = getattr(pg, f"{event_type}")
            self.mouse_value_binds[event_type_value] = mouse_binds

    def SetAttributes(self):
        self.track_width = 7
        self.save_dir = "Tracks"

        self.key_binds = {
            "KEYDOWN": {
                "q": self.Quit,
                "ESCAPE": self.Quit,
                "s": self.Save,
                "r": self.ResetTrack,
            }
        }

        self.mouse_binds = {
            "MOUSEBUTTONDOWN": {
                1: self.LeftMouseButtonDown,
                4: self.ZoomIn,
                5: self.ZoomOut,
            },
            "MOUSEBUTTONUP": {
                1: [self.AddPoint, self.LeftMouseButtonUp],
                3: self.DeletePoint,
            },
        }

        self.zoom = 5.0
        self.zoom_factor = 1.1
        self.zoom_move_factor = 0.1
        self.min_zoom_reached = False
        self.max_zoom_reached = False
        self.min_zoom = 2
        self.max_zoom = 50

        self.cursor_radius = 5.0
        self.way_point_radius = 3.0
        self.rail_handle_radius = 1.5

        self.background_color = (0, 0, 0)
        self.track_color = (5, 5, 5)
        self.rails_color = (255, 255, 255)
        self.way_point_color = (255, 255, 0)
        self.way_point_hover_color = (255, 255, 255)
        self.rail_handle_color = (255, 255, 0)
        self.rail_handle_hover_color = (255, 255, 255)
        self.cursor_color = (255, 255, 0)

        self.start_line_color = (255, 255, 0)

        self.left_mouse_button = False

        self.point_hover_index = -1
        self.point_hover_type = -1

        self.drag_index = -1
        self.drag_type = -1

        self.is_saving = False

        self.dtype = T.float64
        self.device = "cpu"

    def ZoomIn(self):
        zoom = self.zoom * self.zoom_factor

        if zoom > self.max_zoom:
            zoom = self.max_zoom
            self.max_zoom_reached = True
        else:
            self.max_zoom_reached = False

        self.zoom = zoom

        if not self.max_zoom_reached:
            self.screen_pos = (
                (1 - self.zoom_move_factor) * self.screen_pos[0]
                + self.zoom_move_factor * self.global_mouse_pos[0],
                (1 - self.zoom_move_factor) * self.screen_pos[1]
                + self.zoom_move_factor * self.global_mouse_pos[1],
            )

    def ZoomOut(self):
        zoom = self.zoom / self.zoom_factor

        if zoom < self.min_zoom:
            zoom = self.min_zoom
            self.min_zoom_reached = True

        else:
            self.min_zoom_reached = False

        self.zoom = zoom

        if not self.min_zoom_reached:
            self.screen_pos = (
                (1 + self.zoom_move_factor) * self.screen_pos[0]
                - self.zoom_move_factor * self.global_mouse_pos[0],
                (1 + self.zoom_move_factor) * self.screen_pos[1]
                - self.zoom_move_factor * self.global_mouse_pos[1],
            )

    def AddPoint(self):
        if (
            self.n_points > 0
            and self.global_mouse_pos == self.points[-1]
            or self.is_closed
            or self.point_hover_index > 0
            or self.point_hover_index == 0
            and self.n_points < 3
        ):
            return

        if self.point_hover_index == 0 and self.n_points >= 3:
            self.is_closed = True

        else:
            self.points.append(self.global_mouse_pos)
            self.edited_rails.append(False)
            self.n_points += 1

        self.CalculateRails()

    def DeletePoint(self):
        if self.is_closed:
            self.background_color = (0, 0, 0)

            if self.point_hover_index == -1 and self.point_hover_type == -1:
                self.is_closed = False

                self.edited_rails[0] = False
                self.edited_rails[-1] = False

            elif self.point_hover_type == 0:
                self.points.pop(self.point_hover_index)
                self.left_rails.pop(self.point_hover_index)
                self.right_rails.pop(self.point_hover_index)
                self.edited_rails.pop(self.point_hover_index)

                self.n_points -= 1

                if self.n_points == 2:
                    self.is_closed = False

                    self.edited_rails[0] = False
                    self.edited_rails[1] = False

            self.CalculateRails()

        elif self.n_points > 0 and self.point_hover_type <= 0:
            self.background_color = (0, 0, 0)

            self.points.pop(self.point_hover_index)
            self.left_rails.pop(self.point_hover_index)
            self.right_rails.pop(self.point_hover_index)
            self.edited_rails.pop(self.point_hover_index)

            if self.point_hover_index == 0:
                self.edited_rails[0] = False

            elif self.n_points > 1 and (
                self.point_hover_index == self.n_points - 1
                or self.point_hover_index == -1
            ):
                self.edited_rails[-1] = False

            self.n_points -= 1

            self.CalculateRails()

    def DragPoint(self):
        if self.drag_index >= self.n_points:
            print("Fixed")
            return

        if self.drag_index != -1:
            self.background_color = (0, 0, 0)

            if self.drag_type == 0:
                if self.edited_rails[self.drag_index]:
                    self.left_rails[self.drag_index] = (
                        self.left_rails[self.drag_index][0]
                        + self.global_mouse_pos[0]
                        - self.points[self.drag_index][0],
                        self.left_rails[self.drag_index][1]
                        + self.global_mouse_pos[1]
                        - self.points[self.drag_index][1],
                    )
                    self.right_rails[self.drag_index] = (
                        self.right_rails[self.drag_index][0]
                        + self.global_mouse_pos[0]
                        - self.points[self.drag_index][0],
                        self.right_rails[self.drag_index][1]
                        + self.global_mouse_pos[1]
                        - self.points[self.drag_index][1],
                    )

                self.points[self.drag_index] = self.global_mouse_pos

                self.CalculateRails()

            elif self.drag_type == 1:
                self.left_rails[self.drag_index] = self.global_mouse_pos
                self.points[self.drag_index] = (
                    (self.global_mouse_pos[0] + self.right_rails[self.drag_index][0])
                    / 2,
                    (self.global_mouse_pos[1] + self.right_rails[self.drag_index][1])
                    / 2,
                )

                self.edited_rails[self.drag_index] = True

            elif self.drag_type == 2:
                self.right_rails[self.drag_index] = self.global_mouse_pos
                self.points[self.drag_index] = (
                    (self.global_mouse_pos[0] + self.left_rails[self.drag_index][0])
                    / 2,
                    (self.global_mouse_pos[1] + self.left_rails[self.drag_index][1])
                    / 2,
                )

                self.edited_rails[self.drag_index] = True

    def ResetTrack(self):
        self.points = []
        self.n_points = 0

        self.left_rails = []
        self.right_rails = []

        self.is_closed = False

        self.edited_rails = []

    def LeftMouseButtonDown(self):
        self.left_mouse_button = True
        self.drag_index = self.point_hover_index
        self.drag_type = self.point_hover_type

    def LeftMouseButtonUp(self):
        self.left_mouse_button = False
        self.drag_index = -1
        self.drag_type = -1

    def CheckKeyEvents(self, event: pg.event.Event):
        for event_type, key_binds in self.key_value_binds.items():
            if event.type == event_type:
                for key, methods in key_binds.items():
                    if event.key == key:
                        if isinstance(methods, list):
                            for method in methods:
                                method()

                        else:
                            methods()

    def CheckMouseEvents(self, event: pg.event.Event):
        for event_type, mouse_binds in self.mouse_value_binds.items():
            if event.type == event_type:
                for key, methods in mouse_binds.items():
                    if event.button == key:
                        if isinstance(methods, list):
                            for method in methods:
                                method()

                        else:
                            methods()

    def CheckPointHover(self):
        if self.n_points == 0:
            self.point_hover_index = -1
            self.point_hover_type = -1

            return

        if self.n_points == 1:
            min_distance = (
                (self.global_mouse_pos[0] - self.points[0][0]) ** 2
                + (self.global_mouse_pos[1] - self.points[0][1]) ** 2
            ) ** 0.5

            if min_distance <= self.way_point_radius:
                self.point_hover_index = 0
                self.point_hover_type = 0

            else:
                self.point_hover_index = -1
                self.point_hover_type = -1

            return

        min_distance_index = 0
        min_distance_type = 0
        min_distance = (
            (self.global_mouse_pos[0] - self.points[0][0]) ** 2
            + (self.global_mouse_pos[1] - self.points[0][1]) ** 2
        ) ** 0.5

        for point_index, point in enumerate(self.points[1:], start=1):
            distance = (
                (self.global_mouse_pos[0] - point[0]) ** 2
                + (self.global_mouse_pos[1] - point[1]) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance_index = point_index
                min_distance_type = 0
                min_distance = distance

        rail_start_index = 0 if self.is_closed else 1
        rail_stop_index = self.n_points if self.is_closed else self.n_points - 1

        for point_index, point in enumerate(
            self.left_rails[rail_start_index:rail_stop_index], start=rail_start_index
        ):
            distance = (
                (self.global_mouse_pos[0] - point[0]) ** 2
                + (self.global_mouse_pos[1] - point[1]) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance_index = point_index
                min_distance_type = 1
                min_distance = distance

        for point_index, point in enumerate(
            self.right_rails[rail_start_index:rail_stop_index], start=rail_start_index
        ):
            distance = (
                (self.global_mouse_pos[0] - point[0]) ** 2
                + (self.global_mouse_pos[1] - point[1]) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance_index = point_index
                min_distance_type = 2
                min_distance = distance

        if min_distance_type == 0 and min_distance <= self.way_point_radius:
            self.point_hover_index = min_distance_index
            self.point_hover_type = min_distance_type

        elif min_distance <= self.rail_handle_radius:
            self.point_hover_index = min_distance_index
            self.point_hover_type = min_distance_type

        else:
            self.point_hover_index = -1
            self.point_hover_type = -1

    def UpdateEvents(self):
        self.screen_mouse_pos = pg.mouse.get_pos()
        self.global_mouse_pos = self.ScreenPointToGlobalSpace(self.screen_mouse_pos)

        for event in pg.event.get():
            self.CheckKeyEvents(event)
            self.CheckMouseEvents(event)

    def CalculateRails(self):
        if self.n_points >= 3:
            track_points = np.asarray(self.points, dtype=np.float32)

            if self.is_closed:
                a = np.roll(track_points, 1, axis=0)
                b = track_points
                c = np.roll(track_points, -1, axis=0)

            else:
                a = track_points[:-2]
                b = track_points[1:-1]
                c = track_points[2:]

            ab = b - a
            ab_length = np.sqrt(np.sum(ab**2, axis=-1, keepdims=True))
            ab_p = (
                self.track_width
                * np.concatenate((-ab[:, 1, None], ab[:, 0, None]), axis=-1)
                / ab_length
            )

            bc = c - b
            bc_length = np.sqrt(np.sum(bc**2, axis=-1, keepdims=True))
            bc_p = (
                self.track_width
                * np.concatenate((-bc[:, 1, None], bc[:, 0, None]), axis=-1)
                / bc_length
            )

            abc_M = np.concatenate((ab[..., None], bc[..., None]), axis=-1)

            try:
                t_left = (np.linalg.inv(abc_M) @ (c + bc_p - (a + ab_p))[..., None])[
                    ..., 0, 0, None
                ]
                t_right = (np.linalg.inv(abc_M) @ (c - bc_p - (a - ab_p))[..., None])[
                    ..., 0, 0, None
                ]

            except np.linalg.LinAlgError:
                self.DeletePoint()
                return

            if not self.is_closed:
                left_rails = [self.points[0]]
                right_rails = [self.points[0]]

                for i, (edited_rail, left_rail, right_rail) in enumerate(
                    zip(
                        self.edited_rails[1 : self.n_points - 1],
                        (a + ab_p + ab * t_left).tolist(),
                        (a - ab_p + ab * t_right).tolist(),
                    ),
                    start=1,
                ):
                    if edited_rail:
                        left_rails.append(self.left_rails[i])
                        right_rails.append(self.right_rails[i])

                    else:
                        left_rails.append(left_rail)
                        right_rails.append(right_rail)

                left_rails.append(self.points[-1])
                right_rails.append(self.points[-1])

                self.left_rails = left_rails
                self.right_rails = right_rails

            else:
                left_rails = []
                right_rails = []

                for i, (edited_rail, left_rail, right_rail) in enumerate(
                    zip(
                        self.edited_rails,
                        (a + ab_p + ab * t_left).tolist(),
                        (a - ab_p + ab * t_right).tolist(),
                    )
                ):
                    if edited_rail:
                        left_rails.append(self.left_rails[i])
                        right_rails.append(self.right_rails[i])

                    else:
                        left_rails.append(left_rail)
                        right_rails.append(right_rail)

                self.left_rails = left_rails
                self.right_rails = right_rails
        else:
            self.left_rails = self.points.copy()
            self.right_rails = self.points.copy()

    def GlobalPointsToScreenSpace(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        offset_points = []

        for point in points:
            offset_point = (
                (point[0] - self.screen_pos[0]) * self.zoom + self.screen_pos[0],
                (point[1] - self.screen_pos[1]) * self.zoom + self.screen_pos[1],
            )
            offset_points.append(offset_point)

        return offset_points

    def GlobalPointToScreenSpace(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        return self.GlobalPointsToScreenSpace([point])[0]

    def ScreenPointsToGlobalSpace(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        offset_points = []

        for point in points:
            offset_point = (
                (point[0] - self.screen_pos[0]) / self.zoom + self.screen_pos[0],
                (point[1] - self.screen_pos[1]) / self.zoom + self.screen_pos[1],
            )
            offset_points.append(offset_point)

        return offset_points

    def ScreenPointToGlobalSpace(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        return self.ScreenPointsToGlobalSpace([point])[0]

    def DrawRoad(self):
        for i in range(self.n_points - 1):
            pg.draw.polygon(
                self.screen,
                self.track_color,
                self.GlobalPointsToScreenSpace(
                    [
                        self.left_rails[i],
                        self.left_rails[i + 1],
                        self.right_rails[i + 1],
                        self.right_rails[i],
                    ]
                ),
            )

        if self.is_closed:
            pg.draw.polygon(
                self.screen,
                self.track_color,
                self.GlobalPointsToScreenSpace(
                    [
                        self.left_rails[0],
                        self.left_rails[-1],
                        self.right_rails[-1],
                        self.right_rails[0],
                    ]
                ),
            )

    def DrawRails(self):
        pg.draw.lines(
            self.screen,
            self.rails_color,
            self.is_closed,
            self.GlobalPointsToScreenSpace(self.left_rails),
            width=floor(self.track_width * self.zoom / 4),
        )
        pg.draw.lines(
            self.screen,
            self.rails_color,
            self.is_closed,
            self.GlobalPointsToScreenSpace(self.right_rails),
            width=floor(self.track_width * self.zoom / 4),
        )

    def DrawRailHandles(self):
        start_index = 0 if self.is_closed else 1
        stop_index = self.n_points if self.is_closed else self.n_points - 1

        for i, (l, r) in enumerate(
            zip(
                self.left_rails[start_index:stop_index],
                self.right_rails[start_index:stop_index],
            ),
            start=start_index,
        ):
            l_s = self.GlobalPointToScreenSpace(l)
            r_s = self.GlobalPointToScreenSpace(r)

            radius = self.rail_handle_radius * self.zoom

            if i == self.point_hover_index:
                if self.point_hover_type == 1:
                    pg.draw.circle(
                        self.screen, self.rail_handle_hover_color, l_s, radius
                    )
                    pg.draw.circle(
                        self.screen,
                        self.rail_handle_color,
                        l_s,
                        radius,
                        floor(radius / 3),
                    )

                    pg.draw.circle(self.screen, self.track_color, r_s, radius)
                    pg.draw.circle(
                        self.screen,
                        self.rail_handle_color,
                        r_s,
                        radius,
                        floor(radius / 3),
                    )

                elif self.point_hover_type == 2:
                    pg.draw.circle(self.screen, self.track_color, l_s, radius)
                    pg.draw.circle(
                        self.screen,
                        self.rail_handle_color,
                        l_s,
                        radius,
                        floor(radius / 3),
                    )

                    pg.draw.circle(
                        self.screen, self.rail_handle_hover_color, r_s, radius
                    )
                    pg.draw.circle(
                        self.screen,
                        self.rail_handle_color,
                        r_s,
                        radius,
                        floor(radius / 3),
                    )

                else:
                    pg.draw.circle(self.screen, self.track_color, l_s, radius)
                    pg.draw.circle(
                        self.screen,
                        self.rail_handle_color,
                        l_s,
                        radius,
                        floor(radius / 3),
                    )

                    pg.draw.circle(self.screen, self.track_color, r_s, radius)
                    pg.draw.circle(
                        self.screen,
                        self.rail_handle_color,
                        r_s,
                        radius,
                        floor(radius / 3),
                    )

            else:
                pg.draw.circle(self.screen, self.track_color, l_s, radius)
                pg.draw.circle(
                    self.screen, self.rail_handle_color, l_s, radius, floor(radius / 3)
                )

                pg.draw.circle(self.screen, self.track_color, r_s, radius)
                pg.draw.circle(
                    self.screen, self.rail_handle_color, r_s, radius, floor(radius / 3)
                )

    def DrawWayPoints(self):
        for i, point in enumerate(self.points):
            point_s = self.GlobalPointToScreenSpace(point)
            point_r = self.way_point_radius * self.zoom

            if self.point_hover_type == 0 and i == self.point_hover_index:
                pg.draw.circle(
                    self.screen, self.way_point_hover_color, point_s, point_r
                )
            else:
                pg.draw.circle(self.screen, self.track_color, point_s, point_r)

            pg.draw.circle(
                self.screen, self.way_point_color, point_s, point_r, floor(point_r / 3)
            )

    def DrawTrack(self):
        if self.n_points > 2:
            self.DrawRoad()
        if self.n_points > 1:
            self.DrawRails()

        self.DrawWayPoints()
        self.DrawRailHandles()

    def DrawStartLine(self):
        if self.is_closed:
            left_side = self.GlobalPointToScreenSpace(self.left_rails[0])
            right_side = self.GlobalPointToScreenSpace(self.right_rails[0])

            pg.draw.line(
                self.screen,
                self.start_line_color,
                left_side,
                right_side,
                width=floor(self.track_width * self.zoom / 5),
            )

    def Draw(self):
        self.screen.fill(self.background_color)
        self.DrawTrack()
        pg.draw.circle(
            self.screen, self.cursor_color, self.screen_mouse_pos, self.cursor_radius
        )

    def Run(self):
        while True:
            self.UpdateEvents()
            self.CheckPointHover()
            self.DragPoint()
            self.Draw()
            pg.display.flip()

    def Quit(self):
        if not self.is_saving:
            pg.quit()
            sys.exit()

    def GetTrackName(self) -> str | None:
        current_track_names = os.listdir(self.save_dir)
        current_track_numbers = [
            int(name.split("-")[1]) for name in current_track_names if "-" in name
        ]
        new_track_numbers = list(range(1, len(current_track_numbers) + 2))

        for new_track_number in new_track_numbers:
            if new_track_number not in current_track_numbers:
                return f"Track-{new_track_number}"

    def Save(self):
        if not self.is_closed or self.background_color == (5, 5, 5):
            return

        self.is_saving = True

        left_rails = T.as_tensor(self.left_rails, dtype=T.float32)
        right_rails = T.as_tensor(self.right_rails, dtype=T.float32)
        points = T.as_tensor(self.points, dtype=T.float32)

        left_rails -= points[0]
        right_rails -= points[0]
        points = points - points[0]

        os.makedirs(self.save_dir, exist_ok=True)

        track_name = self.GetTrackName()
        track_save_dir = f"{self.save_dir}//{track_name}"

        os.makedirs(track_save_dir, exist_ok=True)

        T.save(
            {"left_rails": left_rails, "right_rails": right_rails, "points": points},
            f"{track_save_dir}//data.pt",
        )

        self.screen.fill(self.background_color)
        self.DrawRoad()
        self.DrawStartLine()
        self.DrawRails()

        pg.image.save(self.screen, f"{track_save_dir}//preview.png")

        self.background_color = (5, 5, 5)

        self.is_saving = False


if __name__ == "__main__":
    builder = TrackBuilder()
