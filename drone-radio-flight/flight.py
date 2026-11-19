"""Vectorized four-rotor rigid-body simulation in SI units."""
import numpy as np

MASS = 0.45
GRAVITY = 9.81
ARM = 0.13
INERTIA = np.array([0.008, 0.008, 0.014])
YAW_COEFFICIENT = 0.025
# Rotor order: front, right, rear, left.
MIX = np.array([[1, 1, 1, 1], [0, -ARM, 0, ARM],
                [-ARM, 0, ARM, 0], [1, -1, 1, -1]])
INVERSE_MIX = np.linalg.inv(MIX)
DT = 0.025
TICKS = 20


class Flight:
    def __init__(self, starts):
        self.pos = np.array(starts, dtype=float, copy=True)
        self.vel = np.zeros_like(self.pos)
        self.angles = np.zeros_like(self.pos)
        self.rates = np.zeros_like(self.pos)
        self.motors = np.full((*self.pos.shape[:-1], 4), MASS * GRAVITY / 4)

    def update_motors(self, horizontal, altitude):
        roll, pitch, yaw = np.moveaxis(self.angles, -1, 0)
        sy, cy = np.sin(yaw), np.cos(yaw)
        ax, ay = np.moveaxis(np.clip(2.1 * (horizontal - self.vel[..., :2]), -4, 4), -1, 0)
        wanted_roll = np.clip((sy * ax - cy * ay) / GRAVITY, -.39, .39)
        wanted_pitch = np.clip((cy * ax + sy * ay) / GRAVITY, -.39, .39)
        desired = np.stack([wanted_roll, wanted_pitch, np.zeros_like(roll)], -1)
        torque = INERTIA * (45 * (desired - self.angles) - 11 * self.rates)
        vertical = 3.6 * (altitude - self.pos[..., 2]) - 2.9 * self.vel[..., 2]
        thrust = MASS * (GRAVITY + vertical) / np.maximum(np.cos(roll) * np.cos(pitch), .55)
        wrench = np.concatenate([thrust[..., None], torque], axis=-1)
        target = np.clip(wrench @ INVERSE_MIX.T, 0, 3.2)
        self.motors += (target - self.motors) * (DT / .065)

    def advance(self, horizontal, altitude=4.0):
        for _ in range(TICKS):
            self.update_motors(horizontal, altitude)
            roll, pitch, yaw = np.moveaxis(self.angles, -1, 0)
            sr, cr, sp, cp, sy, cy = (np.sin(roll), np.cos(roll), np.sin(pitch),
                                       np.cos(pitch), np.sin(yaw), np.cos(yaw))
            realized = self.motors @ MIX.T
            # Body angular momentum and ZYX Euler-angle kinematics.
            angular = (realized[..., 1:] - np.cross(self.rates, self.rates * INERTIA)) / INERTIA
            self.rates += DT * angular
            p, q, r = np.moveaxis(self.rates, -1, 0)
            combination = sr * q + cr * r
            self.angles += DT * np.stack([p + sp / np.maximum(cp, .2) * combination,
                                         cr * q - sr * r, combination / np.maximum(cp, .2)], -1)
            body_z = np.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], -1)
            acceleration = realized[..., :1] / MASS * body_z - .16 * self.vel
            acceleration[..., 2] -= GRAVITY
            self.vel += DT * acceleration
            self.pos += DT * self.vel
