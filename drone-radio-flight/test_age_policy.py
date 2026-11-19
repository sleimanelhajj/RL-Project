"""Regression checks for received-event age and contextual attenuation."""
import unittest
from types import SimpleNamespace
import numpy as np
from age_experiment_policy import AgePolicy

class AgePolicyTest(unittest.TestCase):
    def test_age_clock_and_drift(self):
        params=[1.,10.,-1.]
        p=AgePolicy(3,params,np.arange(3))
        rows=np.arange(3)
        zeros=np.zeros(3,int)
        w=SimpleNamespace(estimated_drift=np.tile([.18,0],(3,1)),t=10,
                          new_scan_events=(rows,zeros,zeros,zeros,zeros))
        radius=np.ones((3,1,1))
        received=p.teammate_evidence(w,radius,None)[:,0,0]
        expected=np.exp(-10/(10*np.exp(-1.)))
        np.testing.assert_allclose(received,[1.,np.exp(-1.),expected])
        w.t+=1
        w.new_scan_events=tuple(np.empty(0,int) for _ in range(5))
        after=p.teammate_evidence(w,radius,None)[:,0,0]
        np.testing.assert_allclose(after,received*np.exp(-1/np.array([10.,10.,10*np.exp(-1.)])))

if __name__=="__main__":
    unittest.main()
