import torch as T
from LinAlg import LinAlg
from Cars.Skin import Skin


class Car:
    def __init__(
        self,
        car_length: float,
        car_width: float,
        car_height: float,
        car_wheelbase_ratio: float,
        car_track_ratio: float,
        max_wheel_angle: float,
        max_acceleration: float,
        fov: float,
        n_rays: int,
        ray_range: float,
        skin: Skin,
        dtype: T.dtype = T.float64,
        device: str = "cpu",
    ):

        self.car_length = car_length
        self.car_width = car_width
        self.car_height = car_height

        self.edge_width = min(self.car_length, self.car_width) / 10
        self.vision_indicator_radius = 0.2

        self.car_wheelbase = car_length * car_wheelbase_ratio
        self.car_track = car_width * car_track_ratio

        self.max_wheel_angle = max_wheel_angle
        self.max_acceleration = max_acceleration

        self.fov = fov
        self.n_rays = n_rays
        self.ray_range = ray_range

        self.skin = skin

        self.dtype = dtype
        self.device = device

        self.SetLocalCarPoints()
        self.SetLocalRayDirections()

    def SetLocalCarPoints(self):
        half_car_length = self.car_length / 2
        half_car_width = self.car_width / 2

        self.local_car_points = T.as_tensor(
            [
                [half_car_length, half_car_width],
                [-half_car_length, half_car_width],
                [-half_car_length, -half_car_width],
                [half_car_length, -half_car_width],
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def SetLocalRayDirections(self):
        angles = T.linspace(
            -self.fov / 2,
            self.fov / 2,
            self.n_rays,
            dtype=self.dtype,
            device=self.device,
        )

        cos_angles = T.cos(angles)
        sin_angles = T.sin(angles)

        self.local_ray_directions = self.ray_range * T.concat(
            (cos_angles[:, None], sin_angles[:, None]), dim=1
        )

    def UpdateCarRotationMatrix(self):
        self.car_rotaion_matrix = LinAlg.get_rotation_matrix(self.car_angle)

    def UpdateGlobalCarLines(self):
        self.global_car_lines = LinAlg.get_lines(
            self.car_position[None]
            + (self.car_rotaion_matrix @ self.local_car_points[..., None])[..., 0],
            True,
        )

    def UpdateGlobalRayLines(self):
        self.global_ray_lines = T.concat(
            (
                self.car_position[None].repeat(self.n_rays, 1)[..., None],
                self.car_rotaion_matrix @ self.local_ray_directions[..., None],
            ),
            dim=-1,
        )

    def Update(self, track_lines: T.Tensor):
        self.UpdateCarRotationMatrix()
        self.UpdateGlobalCarLines()
        self.UpdateGlobalRayLines()

        self.vision = self.See(track_lines.clone())
        self.crashed = self.Crashed(track_lines.clone())

    def GetCenterOfRotation(self, wheel_angle: T.Tensor) -> T.Tensor:
        tan_wheel_angle = T.tan(wheel_angle)

        center_of_rotation = (
            self.car_rotaion_matrix
            @ T.as_tensor(
                [-self.car_wheelbase / 2, self.car_wheelbase / tan_wheel_angle],
                dtype=self.dtype,
                device=self.device,
            )[..., None]
        )[..., 0]

        return center_of_rotation

    def GetAngleDelta(self, center_of_rotation: T.Tensor, dt: float):
        return (
            self.car_speed
            * dt
            / T.sqrt(T.sum(center_of_rotation**2, dim=0, dtype=self.dtype))
        )

    def RotateAroundCenterOfRoation(
        self, center_of_rotation: T.Tensor, wheel_angle: T.Tensor, dt: float
    ):
        angle_delta = T.sgn(wheel_angle) * self.GetAngleDelta(center_of_rotation, dt)

        self.car_position += (
            center_of_rotation
            - (
                LinAlg.get_rotation_matrix(angle_delta) @ center_of_rotation[..., None]
            )[..., 0]
        )
        self.car_angle += angle_delta

    def Crashed(self, track_lines: T.Tensor) -> T.Tensor:
        return LinAlg.intersecting(self.global_car_lines, track_lines)

    def See(self, track_lines: T.Tensor) -> T.Tensor:
        return LinAlg.get_truncated_depth(self.global_ray_lines, track_lines)

    def Reset(self, position: T.Tensor, car_angle: T.Tensor, track_lines: T.Tensor):
        self.car_angle = car_angle

        self.car_position = position
        self.car_speed = T.as_tensor(0.0, dtype=self.dtype, device=self.device)

        self.Update(track_lines)

    def Step(
        self,
        wheel_angle: T.Tensor,
        acceleration: T.Tensor,
        track_lines: T.Tensor,
        dt: float,
    ):
        self.car_speed += acceleration * self.max_acceleration * dt

        if wheel_angle == 0.0:
            self.car_position += self.car_speed * dt * self.car_rotaion_matrix[:, 0]

        else:
            center_of_roation = self.GetCenterOfRotation(
                wheel_angle * self.max_wheel_angle
            )
            self.RotateAroundCenterOfRoation(
                center_of_roation, wheel_angle * self.max_wheel_angle, dt
            )

        self.Update(track_lines)


if __name__ == "__main__":
    pass
