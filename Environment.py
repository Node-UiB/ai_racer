import sys
import pygame as pg

import torch as T
from Car import Car
from Track import Track
from Camera import Camera
from Reward import Reward

from typing import Tuple
from random import randint


class Environment:
    def __init__(
        self,
        car: Car,
        track_name: str,
        dtype: T.dtype,
        device: str,
        reward_function: Reward,
        render: bool = True,
        random_spawn: bool = False,
        visualize_vision: bool = False,
    ):
        self.car = car
        self.track = Track.Load(track_name, dtype=dtype, device=device)
        self.reward_function = reward_function

        self.dtype = dtype
        self.device = device

        self.render = render
        self.random_spawn = random_spawn
        self.visualize_vision = visualize_vision

        if self.render:
            pg.init()

            display_info = pg.display.Info()
            x_res, y_res = display_info.current_w, display_info.current_h

            self.camera = Camera(
                x_res, y_res, pixel_density=1.0, dtype=dtype, device=device
            )
            self.screen = pg.display.set_mode((x_res, y_res), pg.FULLSCREEN)
            self.clock = pg.time.Clock()

            pg.mouse.set_visible(False)

    def GetSpawn(self) -> tuple:
        if self.random_spawn:
            start_index = randint(0, self.track.points.size(0) - 1)
        else:
            start_index = 0

        start_position = self.track.points[start_index].clone()
        start_angle = T.atan2(
            self.track.track_lines[start_index, 1, 1].clone(),
            self.track.track_lines[start_index, 0, 1].clone(),
        )

        return start_position, start_angle

    def Reset(self) -> Tuple[T.Tensor, bool]:
        spawn_position, spawn_angle = self.GetSpawn()
        self.car.Reset(spawn_position, spawn_angle, self.track.track_lines)

        if self.render:
            self.camera.pixel_density = 1.0
            self.camera.Update(
                self.car.car_position.clone(),
                self.car.car_angle.clone(),
                new_pixel_density=15.0,
            )

        return self.car.See(self.track.track_lines), self.car.Crashed(
            self.track.track_lines
        )

    def DrawTrack(self):
        self.screen.fill((0, 50, 0))

        local_left_rails = (
            self.camera.GlobalToLocalSpace(self.track.left_rails).cpu().numpy()
        )
        local_right_rails = (
            self.camera.GlobalToLocalSpace(self.track.right_rails).cpu().numpy()
        )

        for i in range(local_left_rails.shape[0] - 1):
            pg.draw.polygon(
                self.screen,
                self.track.track_color,
                [
                    local_left_rails[i],
                    local_left_rails[i + 1],
                    local_right_rails[i + 1],
                    local_right_rails[i],
                ],
            )

        pg.draw.polygon(
            self.screen,
            self.track.track_color,
            [
                local_left_rails[0],
                local_left_rails[-1],
                local_right_rails[-1],
                local_right_rails[0],
            ],
        )

        goal_width = round(0.5 * self.camera.pixel_density)
        pg.draw.line(
            self.screen,
            self.track.goal_line_color,
            local_left_rails[0],
            local_right_rails[0],
            width=goal_width,
        )

        rail_width = round(0.2 * self.camera.pixel_density)
        pg.draw.lines(
            self.screen, self.track.rail_color, True, local_left_rails, width=rail_width
        )
        pg.draw.lines(
            self.screen,
            self.track.rail_color,
            True,
            local_right_rails,
            width=rail_width,
        )

    def DrawCar(self):
        if self.visualize_vision:
            local_car_position = (
                self.camera.GlobalToLocalSpace(self.car.car_position[None])[0]
                .cpu()
                .numpy()
            )
            local_ray_lines = (
                self.camera.GlobalToLocalSpace(
                    self.car.global_ray_lines[..., 0]
                    + self.car.vision[:, None] * self.car.global_ray_lines[..., 1]
                )
                .cpu()
                .numpy()
            )
            local_vision_indicator_radius = (
                self.car.vision_indicator_radius * self.camera.pixel_density
            )

            for ray_index in range(self.car.n_rays):
                pg.draw.line(
                    self.screen,
                    self.car.skin.vision_lines_color,
                    local_car_position,
                    local_ray_lines[ray_index],
                )
                pg.draw.circle(
                    self.screen,
                    color=(255, 255, 0),
                    center=local_ray_lines[ray_index],
                    radius=local_vision_indicator_radius,
                )

        local_car_points = (
            self.camera.GlobalToLocalSpace(self.car.global_car_lines[..., 0])
            .cpu()
            .numpy()
        )

        pg.draw.polygon(
            self.screen,
            (
                self.car.skin.car_color
                if not self.car.crashed
                else self.car.skin.crashed_car_color
            ),
            local_car_points,
        )
        pg.draw.lines(
            self.screen,
            self.car.skin.outline_color,
            True,
            local_car_points,
            width=round(self.car.edge_width * self.camera.pixel_density),
        )

    def Quit(self):
        pg.quit()
        sys.exit()

    def Render(self):
        self.DrawTrack()
        self.DrawCar()

        pg.display.flip()

    def Step(
        self, wheel_angle: T.Tensor, acceleration: T.Tensor, dt: float
    ) -> Tuple[T.Tensor, T.Tensor, bool]:
        self.car.Step(wheel_angle, acceleration, self.track.track_lines, dt)

        if self.render:
            self.camera.Update(
                self.car.car_position.clone(),
                self.car.car_angle.clone() + T.pi / 2,
                new_pixel_density=15.0,
            )
            self.Render()
            self.clock.tick(1 / dt)

        reward = self.reward_function(self.car.car_position, self.track, self.car.crashed, dt)

        return self.car.vision.clone(), reward, self.car.crashed
