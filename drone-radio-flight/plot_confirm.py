"""Paired comparisons on the common 300-mission PyBullet test set."""
import numpy as np
import matplotlib.pyplot as plt
from optimize_sweep import OUT

labels={"local-rl":"Local search, trained","untrained":"Local search, initial weights","fixed":"Fixed scanning maneuver","previous":"Previous fitted Q-learning","sweep":"Report-centered sweep"}
colors=["#007969","#D58A33","#526C98","#946CAE","#7D8289"]
data={}
fig,axes=plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={"width_ratios":[1.5,1]})
for (key,name),color in zip(labels.items(),colors):
    a=np.loadtxt(OUT/("confirm-"+key+".csv"),delimiter=",",skiprows=1)
    assert np.array_equal(a[:,0],np.arange(1450000,1450300))
    data[key]=a
    t=np.arange(101)
    axes[0].step(t*.5,100*((a[:,1,None]>0)&(a[:,3,None]<=t)).mean(0),where="post",label=name,color=color)
    print(name,"found/crashes/time capped(s)",a[:,1:3].sum(0),a[:,3].mean()*.5)
axes[0].set(xlim=(0,50),ylim=(0,102),xlabel="Simulated seconds",ylabel="Missions with a detection (%)",title="Detection over time")
axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
bars=axes[1].barh(list(labels.values()),[a[:,1].sum() for a in data.values()],color=colors)
axes[1].bar_label(bars,fmt="%.0f"); axes[1].invert_yaxis()
axes[1].set(xlim=(0,326),xlabel="Targets found out of 300",title="Detections by the deadline")
fig.suptitle("Three-drone search across five radio conditions")
fig.tight_layout(); fig.savefig(OUT/"local-policy-300.png",dpi=180)
rng=np.random.default_rng(292)
groups=np.arange(300).reshape(60,5).T
idx=np.concatenate([rng.choice(g,(10000,60)) for g in groups],axis=1)
for key in list(labels)[1:]:
    d=data["local-rl"]-data[key]
    for j,scale in [(1,100),(3,.5)]:
        print(key,j, np.round([d[:,j].mean()*scale,*np.quantile(d[idx,j].mean(1)*scale,[.025,.975])],2))
