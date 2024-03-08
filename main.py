import torch as T

from Reward import Reward
from Cars.RaceCar import RaceCar
from Environment import Environment
from CarController import CarController

device = "cpu"
dtype = T.float64

track_name = "Track-1"

dt = 1 / 60

render = True
random_spawn = False
visualize_vision = True

car = RaceCar(dtype, device)
car_controller = CarController(dtype, device)
reward_function = Reward(1, 10, 100)
env = Environment(
    car,
    track_name,
    dtype,
    device,
    reward_function,
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

    vision, reward, crashed = env.Step(wheel_angle, acceleration, dt)
