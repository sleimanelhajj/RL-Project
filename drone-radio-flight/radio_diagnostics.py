"""Measure how often radio evidence changes a frozen policy on its factual trajectories."""
import csv
from copy import copy

import numpy as np
from experiment import RADIOS
from flight import Flight
from mission import DRONES, Worlds
from optimize_sweep import OUT
from transit_variant import make


def diagnose(ids, regime):
    world = Worlds(ids, [regime] * len(ids), Flight)
    policy = make(len(ids))
    shadow = make(len(ids))
    # Shallow copy shares flight and report arrays. Only its packet observation differs.
    muted = copy(world)
    muted.remote = np.zeros_like(world.remote)
    counts = np.zeros((len(ids), 4), dtype=int)
    while world.active.any():
        muted.t = world.t
        # Feed executed own scans, including their positions and timing, to both policies.
        # The shadow retains its own waypoint until the next factual decision opportunity.
        shadow.scanned = policy.scanned.copy()
        active = np.broadcast_to(world.active[:, None], shadow.scanned.shape)
        decision = active & (policy.scanned | (world.t == 0))
        actual = policy(world)
        alternative = shadow(muted)
        counts[:, 0] += decision.sum(axis=1)
        counts[:, 1] += (decision & (policy.goal != shadow.goal).any(axis=-1)).sum(axis=1)
        counts[:, 2] += active.sum(axis=1)
        counts[:, 3] += (active & (actual != alternative)).sum(axis=1)
        world.step(actual)
    if regime == "off":
        assert not counts[:, [1, 3]].any()
    return counts


if __name__ == "__main__":
    ids = np.arange(3150000, 3150150)
    with (OUT / "radio-shadow-diagnostics.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["scenario", "radio", "waypoint_decisions", "changed_waypoint",
                         "action_decisions", "changed_action"])
        for regime in RADIOS:
            counts = diagnose(ids, regime)
            writer.writerows((seed, regime, *row) for seed, row in zip(ids, counts))
            waypoint, changed_waypoint, action, changed_action = counts.sum(axis=0)
            print(regime, "waypoint %", round(100 * changed_waypoint / waypoint, 2),
                  "action %", round(100 * changed_action / action, 2),
                  "missions with any change", int((counts[:, 1] > 0).sum()), flush=True)
