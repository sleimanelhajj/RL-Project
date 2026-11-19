"""Select a spatial diversity weight on validation and compare it to recent-two FIFO."""
import csv
import numpy as np
from compact_packets import evaluate, OUT

if __name__=="__main__":
    with (OUT/"compact-validation-4050000.csv").open() as f:
        v=list(csv.DictReader(f))
    assert len(v)==4*120*2
    weights=[.5,2,5]
    utility=[np.mean([float(r["return"])-.008*float(r["seconds"]) for r in v if float(r["novelty_weight"])==a]) for a in weights]
    weight=weights[np.argmax(utility)]
    ids=np.repeat(np.arange(4140000,4140160),2)
    channels=np.tile(["burst25","scattered"],160)
    results=[evaluate(ids,channels,k) for k in [0,weight]]
    with (OUT/"compact-holdout.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["id","radio","weight","seconds","found","crashed","sent","received","evictions","return"])
        for k,a in zip([0,weight],results):
            writer.writerows([i,c,k,*r] for i,c,r in zip(ids,channels,a))
    delta=(results[1]-results[0]).reshape(160,2,-1)
    indices=np.random.default_rng(731).integers(0,160,(10000,160))
    print("validation",utility,"selected",weight)
    for a in results: print("means",a.mean(0))
    print("time by channel",delta[:,:,0].mean(0))
    for j in (0,1,3):
        print("difference",j,delta[:,:,j].mean(),np.quantile(delta[:,:,j].mean(1)[indices].mean(1),[.025,.975]))
