"""Paired PyBullet transfer test of the validation-selected late-recovery policy."""
import sys
import numpy as np
from bullet_flight import BulletFlight
from optimize_sweep import OUT,run

if __name__=="__main__":
    name=sys.argv[1]
    if name=="recovery":
        from late_recovery import make
    elif name=="transit":
        from transit_variant import make
    else:
        raise ValueError(name)
    ids=np.arange(4750000,4750200)
    a=np.vstack([run(g,make(len(g)),BulletFlight) for g in np.array_split(ids,5)])
    np.savetxt(OUT/f"recovery-bullet-{name}.csv",np.column_stack([ids,a]),delimiter=",",header="seed,found,crashed,capped_steps,return",comments="")
    print(name,a.mean(0))
