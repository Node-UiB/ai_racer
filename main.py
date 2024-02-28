import torch as T

from Cars.RaceCar import RaceCar
from CarController import CarController
from Environment import Environment

device = "cuda"
dtype = T.float64

track_name = "Track-1"

dt = 1 / 60

render = True
random_spawn = False
visualize_vision = True

car = RaceCar(dtype, device)
car_controller = CarController(dtype, device)
env = Environment(
    car,
    track_name,
    dtype,
    device,
    render=render,
    random_spawn=random_spawn,
    visualize_vision=visualize_vision,
)

vision, crashed = env.Reset()

while True:
    if crashed:
        vision, crashed = env.Reset()
        continue

    wheel_angle, acceleration, terminate = car_controller.GetActions()

    if terminate:
        env.Quit()

    vision, crashed = env.Step(wheel_angle, acceleration, dt)

