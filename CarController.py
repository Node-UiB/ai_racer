import torch as T
import pygame as pg
from typing import Tuple


class CarController:
    def __init__(self, dtype: T.dtype, device: str):
        self.right = T.as_tensor(1.0, dtype=dtype, device=device)
        self.left = T.as_tensor(-1.0, dtype=dtype, device=device)

        self.zero = T.as_tensor(0.0, dtype=dtype, device=device)

        self.forward = T.as_tensor(1.0, dtype=dtype, device=device)
        self.back = T.as_tensor(-1.0, dtype=dtype, device=device)

        self.acceleration = T.as_tensor(0.0, dtype=dtype, device=device)
        self.wheel_angle = T.as_tensor(0.0, dtype=dtype, device=device)
        self.quit = False

        self.acceleration_mu = 0.9
        self.wheel_angle_mu = 0.9

        self.key_binds = {
            "w": {"state": False, "bind": self.Accelerate},
            "a": {"state": False, "bind": self.TurnLeft},
            "s": {"state": False, "bind": self.Decelerate},
            "d": {"state": False, "bind": self.TurnRight},
            "q": {"state": False, "bind": self.Quit},
            "ESCAPE": {"state": False, "bind": self.Quit},
        }

        self.negative_keybinds = {
            ("w", "s"): {"state": False, "bind": self.IdleAcceleration},
            ("a", "d"): {"state": False, "bind": self.IdleTurn},
        }

    def Quit(self):
        self.quit = True

    def Accelerate(self):
        self.acceleration = (
            self.acceleration_mu * self.acceleration
            + (1 - self.acceleration_mu) * self.forward.clone()
        )

    def IdleAcceleration(self):
        self.acceleration = self.zero.clone()

    def Decelerate(self):
        self.acceleration = (
            self.acceleration_mu * self.acceleration
            + (1 - self.acceleration_mu) * self.back.clone()
        )

    def TurnRight(self):
        self.wheel_angle = (
            self.wheel_angle_mu * self.wheel_angle
            + (1 - self.wheel_angle_mu) * self.right.clone()
        )

    def IdleTurn(self):
        self.wheel_angle = self.zero.clone()

    def TurnLeft(self):
        self.wheel_angle = (
            self.wheel_angle_mu * self.wheel_angle
            + (1 - self.wheel_angle_mu) * self.left.clone()
        )

    def CheckKeys(self):
        for keys in self.negative_keybinds.keys():
            self.negative_keybinds[keys]["state"] = True

        self.quit = False

        for event in pg.event.get(pg.KEYDOWN):
            for key in self.key_binds.keys():
                try:
                    if event.key == getattr(pg, f"K_{key}"):
                        self.key_binds[key]["state"] = True

                except AttributeError:
                    continue

        for event in pg.event.get(pg.KEYUP):
            for key in self.key_binds.keys():
                try:
                    if event.key == getattr(pg, f"K_{key}"):
                        self.key_binds[key]["state"] = False

                        for keys in self.negative_keybinds.keys():
                            if key in keys:
                                self.negative_keybinds[keys]["state"] = False
                                break

                except AttributeError:
                    continue

        for keys in self.negative_keybinds.keys():
            if self.negative_keybinds[keys]["state"] == False:
                for key in keys:
                    try:
                        if self.key_binds[key]["state"] == True:
                            self.negative_keybinds[keys]["state"] = True
                            break

                    except KeyError:
                        continue

    def Update(self):
        for key in self.key_binds.keys():
            if self.key_binds[key]["state"] == True:
                self.key_binds[key]["bind"]()

        for keys in self.negative_keybinds.keys():
            if self.negative_keybinds[keys]["state"] == False:
                self.negative_keybinds[keys]["bind"]()

    def GetActions(self) -> Tuple[T.Tensor, T.Tensor, bool]:
        self.CheckKeys()
        self.Update()

        return self.wheel_angle.clone(), self.acceleration.clone(), self.quit
