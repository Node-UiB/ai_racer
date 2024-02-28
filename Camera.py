import torch as T
from linalg import get_rotation_matrix
from typing import Optional


class Camera:
    def __init__(
        self,
        x_res: int,
        y_res: int,
        pixel_density: float = 20,
        position_mu: float = 0.8,
        angle_mu: float = 0.8,
        pixel_density_mu: float = 0.95,
        dtype: T.dtype = T.float64,
        device: str = "cpu",
    ):

        self.x_res = x_res
        self.y_res = y_res

        self.pixel_density = pixel_density

        self.position_mu = position_mu
        self.angle_mu = angle_mu
        self.pixel_density_mu = pixel_density_mu

        self.dtype = dtype
        self.device = device

        self.position = T.as_tensor([0.0, 0.0], dtype=self.dtype, device=self.device)
        self.angle = T.as_tensor(0.0, dtype=self.dtype, device=self.device)

        self.local_screen_center = T.as_tensor(
            [self.x_res / 2, self.y_res * 4 / 5], dtype=dtype, device=device
        )

    def UpdateRotationMatrix(self):
        self.rotation_matrix = get_rotation_matrix(self.angle)
        self.inv_rotation_matrix = self.rotation_matrix.transpose(1, 0)

    def Update(
        self,
        new_position: T.Tensor,
        new_angle: T.Tensor,
        new_pixel_density: Optional[float] = None,
    ):
        self.position = (
            self.position_mu * self.position + (1 - self.position_mu) * new_position
        )
        self.angle = self.angle_mu * self.angle + (1 - self.angle_mu) * new_angle

        if new_pixel_density is not None:
            self.pixel_density = (
                self.pixel_density_mu * self.pixel_density
                + (1 - self.pixel_density_mu) * new_pixel_density
            )

        self.UpdateRotationMatrix()

    def GlobalToLocalSpace(self, positions: T.Tensor) -> T.Tensor:
        return (
            self.inv_rotation_matrix @ (positions - self.position[None])[..., None]
        )[..., 0] * self.pixel_density + self.local_screen_center[None]

    def LocalToGlobalSpace(self, local_positions: T.Tensor) -> T.Tensor:
        return (
            self.rotation_matrix
            @ (local_positions - self.local_screen_center[None])[..., None]
        )[..., 0] / self.pixel_density + self.position[None]
