"""Compare a validated compact packet with full-ring transmission on the same missions."""
import csv
import numpy as np
from mission import Worlds
from compact_packets import OUT,make

def full_run(ids,channels):
    w=Worlds(ids,channels,event_radio=True)
    p=make(len(ids),"fixed",0)
    while w.active.any(): w.step(p(w))
    return np.column_stack([w.time*.5,w.found,w.crashed,w.sent,w.received,w.evicted_unseen.sum((1,2)),w.returns])

if __name__=="__main__":
    ids=np.repeat(np.arange(4140000,4140160),2)
    channels=np.tile(["burst25","scattered"],160)
    a=full_run(ids,channels)
    np.savetxt(OUT/"compact-full-reference.csv",np.column_stack([ids,np.arange(len(ids))%2,a]),delimiter=",",header="id,radio_index,seconds,found,crashed,sent,received,evictions,return",comments="")
    print("Full means",a.mean(0))
    with (OUT/"compact-holdout.csv").open() as f:
        rows=[r for r in csv.DictReader(f) if float(r["weight"])>0]
    assert np.array_equal(ids,np.array([int(r["id"]) for r in rows]))
    compact=np.array([[float(r[k]) for k in ("seconds","found","crashed","sent","received","evictions","return")] for r in rows])
    diff=(compact-a).reshape(160,2,7).mean(1)
    indices=np.random.default_rng(182).integers(0,160,(10000,160))
    for j in [0,1,3]:
        print("difference",j,diff[:,j].mean(),np.quantile(diff[indices,j].mean(1),[.025,.975]))
