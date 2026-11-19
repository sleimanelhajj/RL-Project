"""Bullet adapter for the search simulator's velocity-to-rotor controller."""
import numpy as np
import pybullet as p
from flight import ARM, DT, GRAVITY, INERTIA, MASS, MIX, TICKS, Flight

PALETTE = ((.14, .80, .95, 1.), (.99, .64, .24, 1.), (.74, .66, 1., 1.))


class BulletFlight(Flight):
    def __init__(self, starts):
        super().__init__(starts)
        self.client = p.connect(p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.contact = np.zeros(len(self.pos), dtype=bool)
        self.offsets = np.zeros_like(self.pos)
        self.offsets[..., 0] = np.arange(len(self.pos))[:, None] * 120
        self.bodies = {}
        self.rotations = np.broadcast_to(np.eye(3), (*self.pos.shape[:-1], 3, 3)).copy()
        try:
            self._spawn()
        except Exception:
            self.close()
            raise

    def _spawn(self):
        client = self.client
        p.setTimeStep(DT, physicsClientId=client)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=client)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=client)
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        p.createMultiBody(0, plane, physicsClientId=client)
        airframe = p.createCollisionShape(p.GEOM_CYLINDER, radius=.24, height=.095, physicsClientId=client)
        visuals = [self._drone_visual(color) for color in PALETTE]
        for index in np.ndindex(self.pos.shape[:-1]):
            position = self.pos[index] + self.offsets[index]
            body = p.createMultiBody(MASS, airframe, visuals[index[1] % len(visuals)], position,
                                     physicsClientId=client)
            p.changeDynamics(body, -1, localInertiaDiagonal=INERTIA.tolist(),
                             linearDamping=0, angularDamping=0, physicsClientId=client)
            self.bodies[index] = body
        self.world_for_body = {body: index[0] for index, body in self.bodies.items()}

    def _drone_visual(self, color):
        # The model is enlarged for the aerial camera. Collision and dynamics use SI geometry.
        arm = ARM * 4
        positions = [[0, 0, 0], [0, 0, 0], [0, 0, .13],
                     [arm, 0, .04], [0, -arm, .04], [-arm, 0, .04], [0, arm, .04]]
        return p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * 3 + [p.GEOM_CYLINDER] * 4,
            halfExtents=[[arm, .08, .08], [.08, arm, .08], [.15, .15, .09]] + [[0, 0, 0]] * 4,
            radii=[0, 0, 0] + [.205] * 4,
            lengths=[0, 0, 0] + [.065] * 4,
            visualFramePositions=positions, rgbaColors=[color] * 7, physicsClientId=self.client)

    def advance(self, horizontal, altitude=4.0):
        self.contact.fill(False)
        client = self.client
        for _ in range(TICKS):
            self.update_motors(horizontal, altitude)
            wrench = self.motors @ MIX.T
            for index, body in self.bodies.items():
                rotation = self.rotations[index]
                force = wrench[index][0] * rotation[:, 2] - MASS * .16 * self.vel[index]
                p.applyExternalForce(body, -1, force, self.pos[index] + self.offsets[index],
                                     p.WORLD_FRAME, physicsClientId=client)
                p.applyExternalTorque(body, -1, wrench[index][1:], p.LINK_FRAME, physicsClientId=client)
            p.stepSimulation(physicsClientId=client)
            for index, body in self.bodies.items():
                position, quaternion = p.getBasePositionAndOrientation(body, physicsClientId=client)
                linear, angular = p.getBaseVelocity(body, physicsClientId=client)
                rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
                self.rotations[index] = rotation
                self.pos[index] = position - self.offsets[index]
                self.angles[index] = p.getEulerFromQuaternion(quaternion)
                self.vel[index] = linear
                self.rates[index] = rotation.T @ angular
            for point in p.getContactPoints(physicsClientId=client):
                for body in point[1:3]:
                    if body in self.world_for_body:
                        self.contact[self.world_for_body[body]] = True

    def close(self):
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1
