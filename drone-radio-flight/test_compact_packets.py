import unittest
import numpy as np
from compact_packets import CompactWorlds
from mission import LINKS

class PacketBudget(unittest.TestCase):
    def test_capacity_and_retransmission(self):
        for rule in ("fifo","novelty"):
            w=CompactWorlds([1],["reliable"],packet_policy=rule)
            w.detectors.fill(1.)
            for cost,event_count in [(8,6),(12,6),(12,6)]:
                old=w.sent.copy()
                w.step(np.full((1,3),8))
                self.assertEqual(int((w.sent-old)[0]),cost*len(LINKS))
                self.assertEqual(len(w.new_scan_events[0]),event_count)
            w.step(np.zeros((1,3),int))
            self.assertEqual(len(w.new_scan_events[0]),0)
    def test_reordering(self):
        w=CompactWorlds([1],["reliable"],packet_policy="novelty",novelty_weight=5.)
        w.detectors.fill(1.)
        w.trace[:3]=True
        for _ in range(3): w.step(np.full((1,3),8))
        w.last_cells[0,:,0]=0
        w.step(np.full((1,3),8))
        self.assertEqual(set(w.new_scan_events[-1]),{1,4})
        w.novelty_weight=0.
        w.step(np.zeros((1,3),int))
        self.assertEqual(set(w.new_scan_events[-1]),{3})
if __name__=="__main__": unittest.main()
