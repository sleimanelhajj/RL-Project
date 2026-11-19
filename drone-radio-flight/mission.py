"""Standalone maritime search mission, message channel and policy inputs."""
import numpy as np
from flight import Flight

DRONES = 3
HORIZON = 100
SIZE = 42
CELL = 3
GRID = SIZE // CELL
ALTITUDE = 4.0
RADIUS = 3.4
SPEED = 3.4
LOSS = 25
COST = .001
angles = np.arange(8) * np.pi / 4
DIRECTIONS = np.vstack([np.stack([np.cos(angles), np.sin(angles)], -1), [0., 0.]])
X, Y = np.meshgrid(np.arange(GRID) * CELL + CELL / 2, np.arange(GRID) * CELL + CELL / 2)
CENTERS = np.stack([X.ravel(), Y.ravel()], -1)
LINKS = [(a, b) for a in range(DRONES) for b in range(DRONES) if a != b]


def radio_trace(seed, kind):
    """Exactly 25 of 100 blocked scheduled slots per directed link."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 701]))
    blocked = np.zeros((HORIZON, len(LINKS)), bool)
    for link in range(len(LINKS)):
        offset = rng.integers(0, HORIZON)
        shuffled = rng.permutation(HORIZON)
        if kind == "scattered":
            blocked[shuffled[:LOSS], link] = True
        elif kind in ("burst5", "burst25"):
            length = 5 if kind == "burst5" else 25
            # Offset prevents all links from failing at the same instant.
            slots = np.concatenate([np.arange(length) + i * (length + 11)
                                    for i in range(LOSS // length)])
            blocked[(offset + slots) % HORIZON, link] = True
        elif kind == "off":
            blocked[:, link] = True
        elif kind != "reliable":
            raise ValueError(kind)
    return blocked


class Worlds:
    def __init__(self, ids, radios, flight_factory=Flight):
        self.n = len(ids)
        self.world = np.arange(self.n)
        self.own = np.zeros((self.n, DRONES, GRID * GRID), dtype=np.int16)
        self.remote = self.own.copy()
        self.team = np.zeros((self.n, GRID * GRID), bool)
        self.starts = np.broadcast_to(np.array([[12., 8., ALTITUDE], [21., 8., ALTITUDE],
                                                 [30., 8., ALTITUDE]]), (self.n, DRONES, 3))
        self.flight = flight_factory(self.starts)
        self.target0 = np.empty((self.n, 2))
        self.drift = np.empty_like(self.target0)
        self.report = np.empty_like(self.target0)
        self.estimated_drift = np.empty_like(self.target0)
        self.detectors = np.empty((HORIZON, self.n, DRONES))
        for i, seed in enumerate(ids):
            rng = np.random.default_rng(int(seed))
            self.report[i] = rng.uniform([11, 14], [31, 30])
            self.target0[i] = np.clip(self.report[i] + rng.normal(0, 6.8, 2), 2, SIZE - 2)
            self.drift[i] = rng.uniform(-.095, .095, 2)
            self.estimated_drift[i] = self.drift[i] + rng.normal(0, .04, 2)
            self.detectors[:, i] = rng.random((HORIZON, DRONES))
        self.trace = np.stack([radio_trace(seed, kind) for seed, kind in zip(ids, radios)], axis=1)
        self.transmit = np.array([kind != "off" for kind in radios])
        self.active = np.ones(self.n, bool)
        self.found = np.zeros(self.n, bool)
        self.crashed = np.zeros(self.n, bool)
        self.time = np.full(self.n, HORIZON)
        self.returns = np.zeros(self.n)
        self.duplicate = np.zeros(self.n, int)
        self.team_stamp = np.zeros((self.n, GRID * GRID), dtype=np.int16)
        self.searches = np.zeros(self.n, int)
        self.sent = np.zeros(self.n, int)
        self.received = np.zeros(self.n, int)
        self.last_cells = np.full((self.n, DRONES, 8), -1, int)
        self.last_stamp = np.zeros_like(self.last_cells)
        self.last_slot = np.zeros((self.n, DRONES), int)
        self.t = 0
        self.rr = self.world[:, None]
        self.aa = np.arange(DRONES)[None, :]

    @staticmethod
    def cell(xy):
        cell = np.clip((xy / CELL).astype(int), 0, GRID - 1)
        return cell[..., 0] + GRID * cell[..., 1]

    def target(self):
        return np.clip(self.target0 + self.drift * self.t, 1., SIZE - 1.)

    def forecast(self):
        return np.clip(self.report + self.estimated_drift * self.t, 0, SIZE)

    def features(self):
        xy = self.flight.pos[..., :2]
        forecast = self.forecast()
        candidates = xy[..., None, :] + DIRECTIONS * 3.1
        cells = self.cell(candidates)
        own = self.own[self.rr[..., None], self.aa[..., None], cells]
        remote = self.remote[self.rr[..., None], self.aa[..., None], cells]
        prior = np.exp(-np.sum((candidates - forecast[:, None, None]) ** 2, -1) / (2 * 8.5 ** 2))
        valid = np.all((candidates > 1) & (candidates < SIZE - 1), -1)
        age = np.where(remote > 0, np.minimum((self.t + 1 - remote) / 32, 1), 0)
        own_evidence = np.where(own > 0, np.exp(-(self.t + 1 - own) / 18), 0)
        sector = np.broadcast_to(np.arange(DRONES)[None, :, None] / 2, (self.n, DRONES, 1))
        base = np.concatenate([xy / SIZE, self.flight.vel[..., :2] / SPEED,
                               (forecast[:, None] - xy) / SIZE,
                               np.full((self.n, DRONES, 1), self.t / HORIZON), sector,
                               prior, valid, own_evidence, remote > 0], -1)
        return np.concatenate([base, age], -1).astype(np.float32)

    def critic(self, obs):
        # Centralized critic uses joint local state. Its inputs omit remote age in every arm.
        core = obs[..., :8].reshape(self.n, -1)
        return np.concatenate([core, self.team.mean(axis=1, keepdims=True)], axis=-1).astype(np.float32)

    def step(self, actions):
        running = self.active.copy()
        command = DIRECTIONS[actions] * SPEED
        xy = self.flight.pos[..., :2]
        # Standard geofence controller shared by all methods.
        command = np.where(((xy < 2) & (command < 0)) | ((xy > SIZE - 2) & (command > 0)), 0, command)
        self.flight.advance(np.where(running[:, None, None], command, 0))
        self.t += 1
        xy = self.flight.pos[..., :2]
        cells = self.cell(xy)
        searching = (actions == 8) & running[:, None]
        recent = self.team_stamp[self.rr, cells]
        self.duplicate += (searching & (recent > 0) & (self.t - recent <= 10)).sum(axis=1)
        self.searches += searching.sum(axis=1)
        rows, drones = np.nonzero(searching)
        if rows.size:
            chosen = cells[rows, drones]
            self.own[rows, drones, chosen] = self.t
            self.team[rows, chosen] = True
            self.team_stamp[rows, chosen] = self.t
            slot = self.last_slot[rows, drones]
            self.last_cells[rows, drones, slot] = chosen
            self.last_stamp[rows, drones, slot] = self.t
            self.last_slot[rows, drones] = (slot + 1) % self.last_cells.shape[-1]
        for link, (src, dst) in enumerate(LINKS):
            stamps = self.last_stamp[:, src]
            count = np.count_nonzero(stamps, axis=-1)
            off = self.trace[self.t - 1, :, link]
            self.sent += np.where(running & self.transmit & (count > 0), 4 + 4 * count, 0)
            deliver = running & ~off
            self.received += np.where(deliver & (count > 0), 4 + 4 * count, 0)
            row, packet = np.nonzero(deliver[:, None] & (stamps > 0))
            np.maximum.at(self.remote[:, dst], (row, self.last_cells[row, src, packet]), stamps[row, packet])
        distance = np.linalg.norm(xy - self.target()[:, None], axis=-1)
        detected = np.any(searching & (distance <= RADIUS) & (self.detectors[self.t - 1] < .75), axis=1)
        contact = getattr(self.flight, "contact", False)
        crash = running & (contact | np.any(self.flight.pos[..., 2] < .3, axis=1) |
                           np.any((xy < -1) | (xy > SIZE + 1), axis=(1, 2)))
        detected &= ~crash
        reward = -COST * running + detected - crash.astype(float)
        self.returns += reward
        self.time[detected] = self.t
        self.found |= detected
        self.crashed |= crash
        finish = detected | crash | (self.t == HORIZON)
        self.active &= ~finish
        return reward.astype(np.float32), ~self.active


def greedy(world, radio):
    pos = world.flight.pos[..., :2]
    distance = np.linalg.norm(CENTERS - pos[..., None, :], axis=-1)
    forecast = world.forecast()
    variance = 8.5 ** 2 + .11 * world.t
    prior = np.exp(-np.sum((CENTERS - forecast[:, None]) ** 2, -1) / (2 * variance))
    fresh = np.where(world.own > 0, np.exp(-(world.t + 1 - world.own) / 17), 0)
    if radio:
        fresh = np.maximum(fresh, np.where(world.remote > 0, np.exp(-(world.t + 1 - world.remote) / 17), 0))
    # Favor the preassigned sector without forbidding useful handoffs.
    sector = np.minimum((CENTERS[:, 0] / (SIZE / DRONES)).astype(int), DRONES - 1)
    priority = np.where(sector == np.arange(DRONES)[:, None], 1, .47 if radio else 0)
    score = prior[:, None] * (1 - .93 * fresh) * priority / (1 + .075 * distance)
    goal = CENTERS[score.argmax(-1)]
    return steer(pos, goal)


def steer(position, goal):
    vec = goal - position
    direction = (vec @ DIRECTIONS[:8].T).argmax(-1)
    return np.where(np.linalg.norm(vec, axis=-1) < 1.85, 8, direction)


class Sweep:
    def __init__(self, n):
        self.next = np.zeros((n, DRONES), int)
        self.route = np.array([[(x + 14 * a, y) for iy, y in enumerate((8, 13, 18, 23, 28, 33, 38))
                                for x in ((3, 8, 12) if iy % 2 == 0 else (12, 8, 3))]
                               for a in range(DRONES)])

    def __call__(self, world):
        actions = steer(world.flight.pos[..., :2], self.route[np.arange(DRONES), self.next])
        self.next = (self.next + (actions == 8)) % self.route.shape[1]
        return actions
