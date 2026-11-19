"""Regression checks for event identity and byte accounting on the eight-scan radio."""
import unittest

import numpy as np

from mission import DRONES, Worlds


class Stationary:
    def __init__(self, starts):
        self.pos = np.array(starts)
        self.vel = np.zeros_like(self.pos)

    def advance(self, command):
        pass


def mission(channel, events=True):
    w = Worlds([123], [channel], Stationary, event_radio=events)
    w.detectors.fill(1)
    return w


class RadioEventsTest(unittest.TestCase):
    def test_recovery_and_retransmission(self):
        w = mission("reliable")
        w.trace[:3] = True
        for _ in range(4):
            w.step(np.full((1, DRONES), 8))
        rows, dst, src, cell, tick = w.new_scan_events
        mask = (dst == 1) & (src == 0)
        np.testing.assert_array_equal(np.sort(tick[mask]), [1, 2, 3, 4])
        self.assertEqual(len(np.unique(cell[mask])), 1)
        self.assertEqual(len(rows), 24)
        self.assertEqual(w.received[0], 6 * (4 + 4 * 4))
        w.step(np.zeros((1, DRONES), int))
        self.assertEqual(len(w.new_scan_events[0]), 0)
        self.assertEqual(w.received[0], 2 * 6 * (4 + 4 * 4))

    def test_capacity_and_off(self):
        w = mission("reliable")
        w.trace[:10] = True
        off = mission("off")
        for _ in range(11):
            for active in (w, off):
                active.step(np.array([[8, 0, 0]]))
        _, dst, src, _, tick = w.new_scan_events
        np.testing.assert_array_equal(np.sort(tick[(dst == 1) & (src == 0)]), np.arange(4, 12))
        np.testing.assert_array_equal(w.evicted_unseen[0, 0], [0, 3, 3])
        self.assertEqual(len(off.new_scan_events[0]), 0)
        self.assertEqual(off.sent[0] + off.received[0], 0)

    def test_legacy_byte_and_map_parity(self):
        old, new = mission("scattered", False), mission("scattered")
        rng = np.random.default_rng(23)
        for _ in range(35):
            actions = rng.choice([0, 8], size=(1, DRONES))
            old.step(actions)
            new.step(actions)
            for field in ("remote", "received", "sent", "returns"):
                np.testing.assert_array_equal(getattr(old, field), getattr(new, field))


if __name__ == "__main__":
    unittest.main()
