import torch as T

from linalg import *
from Cars.Skin import Skin

class Car:
    def __init__(self,
                 car_length : float,
                 car_width : float,
                 car_height : float,
                 car_wheelbase_ratio : float,
                 car_track_ratio : float,
                 max_wheel_angle : float,
                 max_acceleration : float,
                 fov : float,
                 n_rays : float,
                 ray_range : float,
                 skin : Skin,
                 dtype : T.dtype = T.float64,
                 device : str = "cpu"):
        
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

        self.local_car_points = T.as_tensor([[half_car_length, half_car_width], 
                                             [-half_car_length, half_car_width], 
                                             [-half_car_length, -half_car_width], 
                                             [half_car_length, -half_car_width]],
                                             dtype=self.dtype,
                                             device=self.device)
        

    def SetLocalRayDirections(self):
        angles = T.linspace(-self.fov / 2, self.fov / 2, self.n_rays, dtype=self.dtype, device=self.device)

        cos_angles = T.cos(angles)
        sin_angles = T.sin(angles)

        self.local_ray_directions = self.ray_range * T.concat((cos_angles[:, None], sin_angles[:, None]), dim=1)
        

    def UpdateCarRotationMatrix(self):
        self.car_rotaion_matrix = get_rotation_matrix(self.car_angle)


    def UpdateGlobalCarLines(self):
        self.global_car_lines = get_lines(self.car_position[None] + (self.car_rotaion_matrix @ self.local_car_points[..., None])[..., 0], True)


    def UpdateGlobalRayLines(self):
        self.global_ray_lines = T.concat((self.car_position[None].repeat(self.n_rays, 1)[..., None], self.car_rotaion_matrix @ self.local_ray_directions[..., None]), dim=-1)

    
    def Update(self, track_lines : T.Tensor):
        self.UpdateCarRotationMatrix()
        self.UpdateGlobalCarLines()
        self.UpdateGlobalRayLines()

        self.vision = self.See(track_lines.clone())
        self.crashed = self.Crashed(track_lines.clone())


    def GetCenterOfRotation(self, wheel_angle : T.Tensor) -> T.Tensor:
        tan_wheel_angle = T.tan(wheel_angle)

        center_of_rotation = (self.car_rotaion_matrix @ T.as_tensor([-self.car_wheelbase / 2, self.car_wheelbase / tan_wheel_angle], dtype=self.dtype, device=self.device)[..., None])[..., 0]

        return center_of_rotation
    

    def GetAngleDelta(self, center_of_rotation : T.Tensor, dt : float):
        return self.car_speed * dt / T.sqrt(T.sum(center_of_rotation ** 2, dim=0, dtype=self.dtype))


    def RotateAroundCenterOfRoation(self, center_of_rotation : T.Tensor, wheel_angle : T.Tensor, dt : float):
        angle_delta = T.sgn(wheel_angle) * self.GetAngleDelta(center_of_rotation, dt)

        self.car_position += center_of_rotation - (get_rotation_matrix(angle_delta) @ center_of_rotation[..., None])[..., 0]
        self.car_angle += angle_delta


    def Crashed(self, track_lines : T.Tensor) -> bool:
        return intersecting(self.global_car_lines, track_lines)


    def See(self, track_lines : T.Tensor) -> T.Tensor:
        return get_truncated_depth(self.global_ray_lines, track_lines)


    def Reset(self, position : T.Tensor, car_angle : T.Tensor, track_lines : T.Tensor):
        self.car_angle = car_angle

        self.car_position = position
        self.car_speed = T.as_tensor(0.0, dtype=self.dtype, device=self.device)

        self.Update(track_lines)
    

    def Step(self, wheel_angle : T.Tensor, acceleration : T.Tensor, track_lines : T.Tensor, dt : float):
        self.car_speed += acceleration * self.max_acceleration * dt

        if(wheel_angle == 0.0):
            self.car_position += self.car_speed * dt * self.car_rotaion_matrix[:, 0]

        else:
            center_of_roation = self.GetCenterOfRotation(wheel_angle * self.max_wheel_angle)
            self.RotateAroundCenterOfRoation(center_of_roation, wheel_angle * self.max_wheel_angle, dt)

        self.Update(track_lines)

    

if(__name__ == "__main__"):
    import sys
    import pygame

    from time import time

    device = "cpu"
    dtype = T.float64

    track_points = T.as_tensor([[410, 410], [1510, 410], [1510, 670], [410, 670]], dtype=dtype, device=device)
    track_lines = get_lines(track_points, True)

    car_position = T.as_tensor([1300.0, 100.0], dtype=dtype, device=device)
    car_angle = T.as_tensor(0.0, dtype=dtype, device=device)

    car_length = 100
    car_width = 50
    car_height = 50

    car_wheelbase_ratio = 0.9
    car_track_ratio = 0.9

    max_wheel_angle = T.pi / 8
    max_acceleration = 100.0

    fov = 2 * T.pi / 3
    n_rays = 5
    ray_range = 1000.0
    
    car = Car(car_length,
              car_width,
              car_height,
              car_wheelbase_ratio,
              car_track_ratio,
              max_wheel_angle,
              max_acceleration,
              fov,
              n_rays,
              ray_range,
              dtype,
              device)
    
    vision, crashed = car.Reset(car_position, car_angle, track_lines.clone())
    
    pygame.init()

    WIDTH, HEIGHT = 1920, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    FORWARD = T.as_tensor(1.0, dtype=dtype, device=device)
    BACKWARD = T.as_tensor(-1.0, dtype=dtype, device=device)
    RIGHT = T.as_tensor(1.0, dtype=dtype, device=device)
    LEFT = T.as_tensor(-1.0, dtype=dtype, device=device)
    
    t0 = time()
    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        acceleration = T.as_tensor(0.0, dtype=dtype, device=device)
        wheel_angle = T.as_tensor(0.0, dtype=dtype, device=device)

        if(keys[pygame.K_ESCAPE] or keys[pygame.K_q]):
            pygame.quit()
            sys.exit()

        if keys[pygame.K_w]:
            acceleration += FORWARD

        if keys[pygame.K_s]:
            acceleration += BACKWARD

        if keys[pygame.K_a]:
            wheel_angle += LEFT
            
        if keys[pygame.K_d]:
            wheel_angle += RIGHT

        tf = time()
        dt = tf - t0
        t0 = tf

        depths, crashed = car.Step(wheel_angle, acceleration, track_lines.clone(), dt)

        car_points = car.global_car_lines[:, :, 0].cpu().numpy().astype(int)

        for i in range(n_rays):
            pygame.draw.line(screen, (255, 255, 255), car.car_position.cpu().numpy(), (car.global_ray_lines[i, :, 0] + depths[i, None] * car.global_ray_lines[i, :, 1]).cpu().numpy())

        pygame.draw.polygon(screen, WHITE if crashed == False else (255, 0, 0), car_points)

        pygame.draw.circle(screen, (255, 255, 0), car_points[0], 3)
        pygame.draw.circle(screen, (255, 255, 0), car_points[-1], 3)
        pygame.draw.circle(screen, (255, 0, 0), car_points[1], 3)
        pygame.draw.circle(screen, (255, 0, 0), car_points[2], 3)

        pygame.draw.lines(screen, (255, 255, 0), True, track_points.cpu().numpy(), width=2)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()