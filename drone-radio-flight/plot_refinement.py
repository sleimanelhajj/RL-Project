import sys
import matplotlib.pyplot as plt
import numpy as np
from optimize_sweep import OUT

def summary(selected):
    names=[selected,"prior","fixed","control"]
    labels=["Selected RL policy","Previous RL policy","Fixed scanning maneuver","New design before training"]
    rng=np.random.default_rng(552)
    idx=np.concatenate([rng.choice(np.arange(i,300,5),(12000,60)) for i in range(5)],axis=1)
    fig,ax=plt.subplots(figsize=(8,4.5))
    records=[]
    for name,label in zip(names,labels):
        a=np.loadtxt(OUT/("optimization-"+name+".csv"),delimiter=",",skiprows=1)
        assert np.array_equal(a[:,0],np.arange(1960000,1960300))
        records.append(a)
        t=np.arange(101)
        ax.step(t*.5,100*((a[:,1,None]>0)&(a[:,3,None]<=t)).mean(0),where="post",label=label)
        print(name,a[:,1:3].sum(0),a[:,3].mean()*.5,"found by 10 and 20s",[((a[:,1]>0) & (a[:,3]<=s)).sum() for s in (20,40)])
    for a in records[1:]:
        d=records[0]-a
        for j,scale in [(1,100),(3,.5)]:
            print(j,np.quantile(d[idx,j].mean(1)*scale,[.025,.975]))
    ax.set(title="Three-drone search on 300 fresh matched missions",xlabel="Simulated seconds",ylabel="Missions with a detection (%)",xlim=(0,50),ylim=(0,101))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"refinement-300.png",dpi=180)
if __name__=="__main__":
    summary(sys.argv[1])


