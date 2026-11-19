"""Train a one-step scan confirmation decision near the forecast after a miss."""
import numpy as np
from mission import DIRECTIONS
from flight import Flight
from transit_variant import Transit,make as transit
from optimize_sweep import OUT,run

class ConfirmMiss(Transit):
    def __init__(self,n,weights):
        super().__init__(n,np.load(OUT/"transit-weights.npy"))
        self.params=np.broadcast_to(weights,(n,3))
        self.cooldown=np.zeros((n,3),int)

    def __call__(self,w):
        previous=self.scanned.copy()
        action=super().__call__(w)
        # A second scan is useful only if the previous one missed and its footprint overlaps.
        xy=w.flight.pos[...,:2]
        dist=np.linalg.norm(xy-w.forecast()[:,None],axis=-1)
        speed=np.linalg.norm(w.flight.vel[...,:2],axis=-1)
        radius,uncertainty_cost,spacing=self.params.T
        ids=w.cell(xy)
        evidence=np.take_along_axis(self.log_evidence,ids[:,:,None],axis=-1)[:,:,0]
        # One unit of evidence is expected from the just-completed scan.
        repeat=(previous & (action!=8) & (self.cooldown==0) &
                (dist+uncertainty_cost[:,None]*np.maximum(evidence-1,0)+speed < radius[:,None]))
        self.cooldown=np.where(repeat,spacing[:,None].astype(int),np.maximum(0,self.cooldown-1))
        action[repeat]=8
        self.scanned=action==8
        return action

def objective(a):
    return (a[...,3]-.004*a[...,2]).mean(-1)

def make(n):
    return ConfirmMiss(n,np.load(OUT/"confirm-miss-weights.npy"))

if __name__=="__main__":
    rng=np.random.default_rng(9060)
    mean=np.array([6.,4.,5.])
    spread=np.array([4.,2.5,3.])
    candidates=[mean.copy(),np.array([-1.,0.,0.])]
    for g in range(3):
        p=np.clip(rng.normal(mean,spread,(6,3)),[1.,0.,1.],[14.,12.,20.])
        ids=np.arange(4710000+g*100,4710040+g*100)
        a=run(np.tile(ids,len(p)),ConfirmMiss(len(ids)*len(p),np.repeat(p,len(ids),axis=0)),Flight)
        elite=p[np.argsort(objective(a.reshape(len(p),len(ids),4)))[-2:]]
        mean=elite.mean(0); spread=np.maximum(elite.std(0),spread*.6)
        candidates.append(mean.copy())
    ids=np.arange(4720000,4720200)
    evals=[run(ids,ConfirmMiss(len(ids),p),Flight) for p in candidates]
    scores=[objective(a) for a in evals]
    best=np.argmax(scores)
    np.save(OUT/"confirm-miss-weights.npy",candidates[best])
    print("validation",scores,"choice",best,candidates[best])
    print("baseline vs selected",evals[1].mean(0),evals[best].mean(0))
