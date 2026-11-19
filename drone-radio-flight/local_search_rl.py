"""Episodic cross-entropy policy search for local scan-waypoint decisions."""
import numpy as np
from mission import CENTERS, steer
from flight import Flight
from optimize_sweep import run, OUT
from confirm_search import evaluate

INITIAL=np.array([7.,3.3,2.4,.12,4.6,55.])
LOW=np.array([4.,2.,.7,.015,3.,20.])
HIGH=np.array([15.,5.,5.,.36,8.,140.])

class LocalSearch:
    def __init__(self,n,weights):
        self.weights=np.broadcast_to(weights,(n,6)).T[:,:,None,None]
        self.log_evidence=np.zeros((n,3,len(CENTERS)))
        self.goal=np.zeros((n,3,2))
        self.scanned=np.zeros((n,3),bool)
        self.last_remote=np.zeros((n,3,len(CENTERS)),dtype=np.int16)

    def waypoint_score(self, w, log_mass, distance, travel):
        return log_mass - travel * distance

    def __call__(self,w):
        scale,radius,penalty,travel,width,decay=self.weights
        xy=w.flight.pos[...,:2]
        d2=((xy[:,:,None]-CENTERS)**2).sum(-1)
        self.log_evidence *= np.exp(-1/decay)
        self.log_evidence += self.scanned[:,:,None]*np.exp(-d2/(2*radius**2))
        # Received scan timestamps are the only source of teammate evidence.
        rows,agents,cells=np.nonzero(w.remote>self.last_remote)
        if len(rows):
            delta=CENTERS[cells,None]-CENTERS
            kernel=np.exp(-(delta**2).sum(-1)/(2*radius[rows,0,0]**2)[:,None])
            age=w.t-w.remote[rows,agents,cells]
            kernel*=np.exp(-age/decay[rows,0,0])[:,None]
            np.add.at(self.log_evidence,(rows,agents),kernel)
        self.last_remote[:]=w.remote
        center=w.forecast()[:,None,None,:]
        prior=((CENTERS-center)**2).sum(-1)/(2*scale**2)
        score=self.waypoint_score(w, -prior-penalty*self.log_evidence, np.sqrt(d2), travel)
        left=w.report[:,0,None,None]-width
        right=w.report[:,0,None,None]+width
        x=CENTERS[:,0]
        sector=np.arange(3)[None,:,None]
        valid=np.where(sector==0,x<left,np.where(sector==1,(x>=left)&(x<=right),x>right))
        best=CENTERS[np.where(valid,score,-np.inf).argmax(-1)]
        self.goal=np.where((self.scanned|(w.t==0))[:,:,None],best,self.goal)
        action=steer(xy,self.goal)
        self.scanned=action==8
        return action

def train():
    rng=np.random.default_rng(410)
    mean=INITIAL.copy()
    spread=(HIGH-LOW)*.25
    finalists=[mean.copy()]
    for generation in range(4):
        candidates=np.clip(rng.normal(mean,spread,(10,6)),LOW,HIGH)
        candidates[0]=mean
        seeds=np.arange(1200000+generation*100,1200000+generation*100+35)
        ids=np.tile(seeds,len(candidates))
        w=np.repeat(candidates,len(seeds),axis=0)
        a=run(ids,LocalSearch(len(ids),w),Flight).reshape(len(candidates),len(seeds),4)
        # Extra time cost favors earlier detection; confirmatory metrics are unshaped.
        score=(a[:,:,3]-.004*a[:,:,2]).mean(1)
        elite=candidates[np.argsort(score)[-3:]]
        mean=elite.mean(0)
        spread=np.maximum(elite.std(0),(HIGH-LOW)*.075)
        finalists.append(mean.copy())
        print("Training generation",generation+1,flush=True)
    ids=np.arange(1350000,1350200)
    scores=[]
    for weights in finalists:
        a=run(ids,LocalSearch(len(ids),weights),Flight)
        scores.append((a[:,3]-.004*a[:,2]).mean())
    best=finalists[np.argmax(scores)]
    np.savez(OUT/"local-search-weights.npz",weights=best,validation=scores,candidates=finalists)
    print("Validation scores",scores,flush=True)
    return best

if __name__=="__main__":
    import sys
    if "control" in sys.argv:
        evaluate("untrained",lambda n:LocalSearch(n,INITIAL))
    else:
        weights=train()
        evaluate("local-rl",lambda n:LocalSearch(n,weights))

