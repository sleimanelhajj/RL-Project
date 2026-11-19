"""Decentralized waypoint policy with missed-scan likelihoods and footprint integration.

Motivated by Bourgault, Furukawa & Durrant-Whyte, ICRA 2004,
https://cornell-asl.org/wiki/images/1/1e/Bourgault04process.pdf
Each aircraft maintains log likelihood ratios for negative observations.
An approximate footprint integral estimates detection utility at each waypoint.
"""
import numpy as np
from mission import CENTERS, DIRECTIONS, RADIUS
from flight import Flight
from optimize_sweep import OUT, run

class BeliefSearch:
    def __init__(self,n,params):
        self.params=np.broadcast_to(params,(n,5)).T[:,:,None,None]
        self.loglike=np.zeros((n,3,196))
        self.scan=np.zeros((n,3),bool)
        self.last=np.full((n,3,2),-100.)
        self.goal=np.zeros_like(self.last)
        self.remote=np.zeros((n,3,196),np.int16)
        self.base=np.load(OUT/"transit-weights.npy")

    def __call__(self,w):
        travel,edge,footprint,memory,power=self.params
        xy=w.flight.pos[...,:2]
        dist=np.linalg.norm(xy[:,:,None]-CENTERS,axis=-1)
        self.last[self.scan]=xy[self.scan]
        self.loglike*=np.exp(-1/memory)
        # Smooth the disk boundary to account for discretization.
        p=.75/(1+np.exp(np.clip((dist-RADIUS)/edge,-60,60)))
        self.loglike+=self.scan[:,:,None]*np.log1p(-p)
        row,agent,cell=np.nonzero(w.remote>self.remote)
        if len(row):
            age=w.t-w.remote[row,agent,cell]
            advected=CENTERS[cell]+age[:,None]*w.estimated_drift[row]
            d=np.linalg.norm(advected[:,None]-CENTERS,axis=-1)
            p=.75/(1+np.exp(np.clip((d-RADIUS)/edge[row,0,:],-60,60)))
            np.add.at(self.loglike,(row,agent),np.log1p(-p)*np.exp(-age[:,None]/memory[row,0,:]))
        self.remote[:]=w.remote
        prior=-np.square(CENTERS-w.forecast()[:,None,:]).sum(-1)/(2*self.base[0]**2)
        mass=np.exp(prior[:,None]+power*self.loglike).reshape(w.n,3,14,14)
        utility=mass.copy()
        # Four cardinal neighbor cells lie within one scan radius.
        weight=footprint[...,None]
        utility[:,:,1:]+=weight*mass[:,:,:-1]
        utility[:,:,:-1]+=weight*mass[:,:,1:]
        utility[:,:,:,1:]+=weight*mass[:,:,:,:-1]
        utility[:,:,:,:-1]+=weight*mass[:,:,:,1:]
        score=np.log(utility.reshape(dist.shape)+1e-30)-travel*dist
        dx=CENTERS[:,0]-w.report[:,0,None]
        valid=np.stack([dx < -self.base[4],abs(dx)<=self.base[4],dx>self.base[4]],axis=1)
        chosen=CENTERS[np.where(valid,score,-np.inf).argmax(-1)]
        self.goal=np.where((self.scan|(w.t==0))[:,:,None],chosen,self.goal)
        coast=xy+w.flight.vel[...,:2]*self.base[6]
        self.scan=(np.linalg.norm(self.goal-coast,axis=-1)<1.85)|((np.linalg.norm(coast-self.last,axis=-1)>self.base[7])&(np.linalg.norm(coast-w.forecast()[:,None],axis=-1)<13))
        return np.where(self.scan,8,((self.goal-xy)@DIRECTIONS[:8].T).argmax(-1))

def make(n):
    return BeliefSearch(n,np.load(OUT/"belief-update.npy"))

if __name__=="__main__":
    rng=np.random.default_rng(341)
    mean=np.array([.16,.65,.3,110,1.4])
    spread=np.array([.065,.3,.24,35,.48])
    finalists=[mean.copy()]
    for k in range(3):
        candidates=np.clip(rng.normal(mean,spread,(8,5)),[.04,.2,.02,30,.4],[.4,1.8,1.2,250,3])
        seeds=np.arange(2381000+100*k,2381035+100*k)
        ids=np.tile(seeds,len(candidates))
        a=run(ids,BeliefSearch(len(ids),np.repeat(candidates,len(seeds),axis=0)),Flight).reshape(len(candidates),len(seeds),4)
        elite=candidates[np.argsort((a[:,:,3]-.004*a[:,:,2]).mean(1))[-3:]]
        mean=elite.mean(0); spread=np.maximum(elite.std(0),spread*.6)
        finalists.append(mean.copy())
        print("Completed generation",k+1,flush=True)
    ids=np.arange(2420000,2420200)
    scores=[]
    for p in finalists:
        a=run(ids,BeliefSearch(len(ids),p),Flight)
        scores.append((a[:,3]-.004*a[:,2]).mean())
        print("Validation",scores[-1],a.mean(0),flush=True)
    np.save(OUT/"belief-update.npy",finalists[np.argmax(scores)])
