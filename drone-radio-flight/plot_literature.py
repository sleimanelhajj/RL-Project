"""Plot the independently selected candidate against two frozen baselines."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from optimize_sweep import OUT

if __name__ == "__main__":
    candidate=sys.argv[1]
    labels={candidate:candidate.replace("_"," ").capitalize(),"transit_variant":"Transit-scanning RL","fixed":"Fixed scan"}
    fig,ax=plt.subplots(figsize=(8,4.7))
    rng=np.random.default_rng(540)
    ix=np.hstack([rng.choice(np.arange(i,300,5),(10000,60)) for i in range(5)])
    first=None
    for key,label in labels.items():
        a=np.loadtxt(OUT/("literature-"+key+".csv"),delimiter=",",skiprows=1)
        assert np.array_equal(a[:,0],np.arange(2520000,2520300))
        print(key,"found, crashes, capped seconds",a[:,1:3].sum(0),a[:,3].mean()*.5)
        if first is None:
            first=a
        else:
            d=first-a
            print("paired intervals",np.quantile(d[ix,1].mean(1)*100,[.025,.975]),np.quantile(d[ix,3].mean(1)*.5,[.025,.975]))
        t=np.arange(101)
        ax.step(t*.5,100*((a[:,1,None]>0)&(a[:,3,None]<=t)).mean(0),where="post",label=label)
    ax.set(xlabel="Simulated mission time (s)",ylabel="Missions with a detection (%)",title="Three-drone search on 300 matched PyBullet missions",xlim=(0,50),ylim=(0,102))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"literature-comparison.png",dpi=180)
