"""Matched comparison of communication and search outcomes in PyBullet."""
import numpy as np
import matplotlib.pyplot as plt
from sender_schedule import OUT

def main():
    names={"full":"8 records each slot","compact":"2 records each slot","once":"Send on new scan","learned":"Trained retry schedule"}
    rng=np.random.default_rng(492)
    idx=rng.integers(0,140,(12000,140))
    vals=[]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.9))
    for key,label in names.items():
        a=np.loadtxt(OUT/f"sender-bullet-{key}.csv",delimiter=",",skiprows=1)
        assert np.array_equal(a[:,0],np.repeat(np.arange(4910000,4910140),2))
        vals.append(a)
        print(key,"found/crashes",a[:,2:4].sum(0),"mean sec, sent, received",a[:,4:7].mean(0))
    for ax,j,title in zip(axes,(5,4),("Bytes transmitted per mission","Mean search time (s)")):
        bars=ax.barh(list(names.values()),[a[:,j].mean() for a in vals],color=["#566b88","#9b8265","#84a3a0","#007568"])
        ax.bar_label(bars,fmt="%.1f",padding=2); ax.invert_yaxis(); ax.margins(x=.17); ax.set_title(title)
    for n,a in zip(names,vals):
        diff=(vals[-1]-a).reshape(140,2,-1).mean(1)
        for j in (2,4,5):
            print(n,j,np.round(np.quantile(diff[idx,j].mean(1),[.025,.975]),3))
    fig.suptitle("140 matched PyBullet scenarios under scattered and burst loss")
    fig.tight_layout(); fig.savefig(OUT/"communication-search.png",dpi=170)
if __name__=="__main__": main()
