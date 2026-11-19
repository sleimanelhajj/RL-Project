import sys
import numpy as np
from optimize_sweep import OUT,run
from bullet_flight import BulletFlight
from local_search_rl import LocalSearch
from sequential_rl import fixed

if __name__=="__main__":
    name=sys.argv[1]
    if name=="prior":
        weights=np.load(OUT/"local-search-weights.npz")["weights"]
        make=lambda n:LocalSearch(n,weights)
    elif name=="fixed":
        make=fixed
    else:
        from importlib import import_module
        make=import_module(name).make
    ids=np.arange(1960000,1960300)
    a=np.vstack([run(g,make(len(g)),BulletFlight) for g in np.array_split(ids,10)])
    np.savetxt(OUT/("optimization-"+name+".csv"),np.column_stack([ids,a]),delimiter=",",header="seed,found,crash,capped_steps,return",comments="")
    print(name,a.mean(0))
