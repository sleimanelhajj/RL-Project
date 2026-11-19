"""Learn waypoint scores on a lattice that follows the estimated target drift."""
import numpy as np
from mission import steer
from flight import Flight
from local_search_rl import LocalSearch
from optimize_sweep import run, OUT

axis = np.arange(-10, 11)*3.
OFFSETS = np.stack(np.meshgrid(axis,axis),axis=-1).reshape(-1,2)

class DriftPolicy:
    def __init__(self,n,weights):
        self.params=np.broadcast_to(weights,(n,6)).T[:,:,None,None]
        self.evidence=np.zeros((n,3,len(OFFSETS)))
        self.goal=np.zeros((n,3,2))
        self.scanned=np.zeros((n,3),bool)
        self.remote=np.zeros((n,3,196),np.int16)

    def __call__(self,w):
        from mission import CENTERS
        scale,radius,penalty,travel,width,drift=self.params
        origin=w.report + w.estimated_drift*w.t*drift[:,0,:]
        grid=origin[:,None,:]+OFFSETS
        xy=w.flight.pos[...,:2]
        delta=grid[:,None]-xy[:,:,None]
        dist=np.sqrt(np.square(delta).sum(-1))
        self.evidence *= .996
        self.evidence += self.scanned[:,:,None]*np.exp(-.5*(dist/radius)**4)
        rows,agents,cells=np.nonzero(w.remote>self.remote)
        if len(rows):
            age=w.t-w.remote[rows,agents,cells]
            scan=CENTERS[cells]+age[:,None]*w.estimated_drift[rows]*drift[rows,0,:]
            d2=np.square(grid[rows]-scan[:,None]).sum(-1)
            np.add.at(self.evidence,(rows,agents),np.exp(-.5*(d2/(radius[rows,0,:]**2))**2)*.996**age[:,None])
        self.remote[:]=w.remote
        prior=np.square(grid-w.forecast()[:,None]).sum(-1)[:,None]/(2*scale**2)
        score=-prior-penalty*self.evidence-travel*dist
        x=OFFSETS[:,0]
        sector=np.arange(3)[None,:,None]
        valid=np.where(sector==0,x < -width,np.where(sector==2,x>width,np.abs(x)<=width))
        valid=valid & ((grid>=.5)&(grid<=41.5)).all(-1)[:,None]
        best=grid[np.arange(w.n)[:,None],np.where(valid,score,-np.inf).argmax(-1)]
        self.goal=np.where((self.scanned|(w.t==0))[:,:,None],best,self.goal)
        action=steer(xy,self.goal)
        self.scanned=action==8
        return action

def make(n):
    return DriftPolicy(n,np.load(OUT/"drift-policy.npy"))

def score(a):
    return (a[...,3]-.004*a[...,2]).mean(axis=-1)

if __name__=="__main__":
    rng=np.random.default_rng(5131)
    mean=np.array([7.4,3.45,1.7,.22,4.3,.8])
    spread=np.array([1.2,.5,.6,.06,1.5,.25])
    candidates=[mean.copy()]
    for g in range(3):
        p=np.clip(rng.normal(mean,spread,(8,6)),[4,2,.5,.04,2,0],[12,5.5,5,.42,8,1.25])
        seeds=np.arange(1810000+100*g,1810032+100*g)
        ids=np.tile(seeds,len(p))
        a=run(ids,DriftPolicy(len(ids),np.repeat(p,len(seeds),axis=0)),Flight)
        elite=p[np.argsort(score(a.reshape(len(p),len(seeds),4)))[-3:]]
        mean=elite.mean(0); candidates.append(mean.copy()); spread*=.78
        print("Generation",g,flush=True)
    ids=np.arange(1850000,1850200)
    scores=[]
    for p in candidates:
        a=run(ids,DriftPolicy(len(ids),p),Flight)
        scores.append(score(a))
        print("Validation",scores[-1],a.mean(0),flush=True)
    np.save(OUT/"drift-policy.npy",candidates[np.argmax(scores)])
    a=run(ids,LocalSearch(len(ids),np.load(OUT/"local-search-weights.npz")["weights"]),Flight)
    print("Frozen",score(a),a.mean(0),flush=True)
