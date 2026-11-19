"""Evaluate the initial parameters of the variant selected using validation."""
import sys
import numpy as np
from optimize_sweep import OUT,run
from bullet_flight import BulletFlight

if __name__=="__main__":
    variant=sys.argv[1]
    base=np.load(OUT/"local-search-weights.npz")["weights"]
    if variant=="drift_variant":
        from drift_variant import DriftPolicy
        make=lambda n:DriftPolicy(n,np.array([7.4,3.45,1.7,.22,4.3,.8]))
    elif variant=="transit_variant":
        from transit_variant import Transit
        make=lambda n:Transit(n,np.r_[base,.48,4.])
    else:
        from phase_variant import Adaptive
        make=lambda n:Adaptive(n,np.tile(base,2))
    ids=np.arange(1960000,1960300)
    a=np.vstack([run(g,make(len(g)),BulletFlight) for g in np.array_split(ids,10)])
    np.savetxt(OUT/"optimization-control.csv",np.column_stack([ids,a]),delimiter=",",header="seed,found,crash,capped_steps,return",comments="")
    print(a.mean(0))
