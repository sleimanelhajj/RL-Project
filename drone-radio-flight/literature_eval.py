"""Run matched PyBullet tests for the literature-based refinements."""
import argparse
from importlib import import_module
import numpy as np
from bullet_flight import BulletFlight
from optimize_sweep import OUT, run
from sequential_rl import fixed

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy",choices=["fixed","transit_variant","belief_update","lookahead_search"])
    name=parser.parse_args().policy
    make=fixed if name=="fixed" else import_module(name).make
    ids=np.arange(2520000,2520300)
    a=np.vstack([run(group,make(len(group)),BulletFlight) for group in np.array_split(ids,10)])
    np.savetxt(OUT/("literature-"+name+".csv"),np.column_stack([ids,a]),delimiter=",",header="seed,found,crash,capped_steps,return",comments="")
    print(name,a.mean(0),flush=True)
