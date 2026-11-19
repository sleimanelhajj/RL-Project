"""Tune how a frozen search policy weighs delayed radio evidence."""
from pathlib import Path
import csv
import sys
import numpy as np
from mission import CENTERS, Worlds
from transit_variant import Transit

OUT=Path(__file__).resolve().parent/"results"
ARMS=("blind","fixed","learned")
CHANNELS=("scattered","burst25")

class AgePolicy(Transit):
    def __init__(self,n,params,arm):
        super().__init__(n,np.load(OUT/"transit-weights.npy"))
        self.arm=np.broadcast_to(arm,(n,))
        params=np.broadcast_to(params,(n,3))
        self.strength,self.memory,self.drift_coef=params.T
        self.evidence=np.zeros_like(self.log_evidence)
        self.mask_reports=False

    def teammate_evidence(self,w,radius,decay):
        if self.mask_reports:
            return 0.
        speed=np.linalg.norm(w.estimated_drift,axis=-1)
        # Speed uses grid distance per tick; tau is measured in ticks.
        memory=self.memory*np.exp(np.clip(self.drift_coef*(speed/.09-1),-2,2)*(self.arm==2))
        self.evidence*=np.exp(-1/memory)[:,None,None]
        rows,dst,_,cells,ticks=w.new_scan_events
        if len(rows):
            age=(w.t-ticks)*(self.arm[rows]!=0)
            diff=CENTERS[cells,None]-CENTERS
            kernel=np.exp(-(diff**2).sum(-1)/(2*radius[rows,0,0]**2)[:,None])
            np.add.at(self.evidence,(rows,dst),kernel*np.exp(-age/memory[rows])[:,None])
        return self.evidence*self.strength[:,None,None]

def make(n,arm,seed):
    arm_id=ARMS.index(arm)
    return AgePolicy(n,np.load(OUT/f"age-policy-{seed}.npy")[arm_id],arm_id)

def score(params,ids,channels):
    arms,k,_=params.shape
    worlds=Worlds(np.tile(ids,arms*k),np.tile(channels,arms*k),event_radio=True)
    policy=AgePolicy(worlds.n,np.repeat(params.reshape(-1,3),len(ids),axis=0),
                     np.repeat(np.arange(arms),k*len(ids)))
    while worlds.active.any():
        worlds.step(policy(worlds))
    return (worlds.returns-.004*worlds.time).reshape(arms,k,-1).mean(-1)

def train(seed):
    rng=np.random.default_rng(318+seed)
    mean=np.tile([1.,45.,0.],(3,1))
    spread=np.array([.65,24.,.8])
    history=[mean.copy()]
    for generation in range(2):
        ids=np.arange(3600000+1000*seed+100*generation,3600000+1000*seed+100*generation+12)
        # Identical proposal noise and missions in each arm for a given training seed.
        noise=rng.normal(size=(5,3))
        candidates=np.clip(mean[:,None]+spread*noise,[.05,4.,-2],[3.,160.,2])
        candidates[:,0]=mean
        scores=score(candidates,ids,np.resize(CHANNELS,len(ids)))
        best=np.argsort(scores,axis=1,kind="stable")[:,-2:]
        mean=candidates[np.arange(3)[:,None],best].mean(1)
        spread*=.65
        history.append(mean.copy())
    history=np.stack(history,axis=1)
    ids=np.arange(3720000,3720032)
    val=score(history,ids,np.resize(CHANNELS,len(ids)))
    winners=val.argmax(1)
    np.save(OUT/f"age-policy-{seed}.npy",history[np.arange(3),winners])
    np.savetxt(OUT/f"age-validation-{seed}.csv",val,delimiter=",",header="initial,generation1,generation2",comments="")
    print(seed,dict(zip(ARMS,val.max(axis=1))),flush=True)

if __name__=="__main__":
    for seed in (map(int,sys.argv[1:]) if sys.argv[1:] else range(5)):
        train(seed)
