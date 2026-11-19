"""Optimize the scanning trigger jointly with waypoint scoring by episodic reward."""
import numpy as np
from mission import DIRECTIONS
from local_search_rl import LocalSearch
from optimize_sweep import OUT, run
from flight import Flight

class Transit(LocalSearch):
    def __init__(self,n,params):
        params=np.broadcast_to(params,(n,8))
        super().__init__(n,params[:,:6])
        self.lead=params[:,6,None,None]
        self.spacing=params[:,7,None]
        self.last=np.full((n,3,2),-100.)
    def __call__(self,w):
        previous=self.scanned
        xy=w.flight.pos[...,:2]
        self.last[previous]=xy[previous]
        super().__call__(w)
        coast=xy+w.flight.vel[...,:2]*self.lead
        direction=self.goal-xy
        action=(direction @ DIRECTIONS[:8].T).argmax(-1)
        scan=np.linalg.norm(self.goal-coast,axis=-1)<1.85
        transit=(np.linalg.norm(coast-self.last,axis=-1)>self.spacing)&(np.linalg.norm(coast-w.forecast()[:,None],axis=-1)<13)
        self.scanned=scan|transit
        return np.where(self.scanned,8,action)

def make(n):
    return Transit(n,np.load(OUT/"transit-weights.npy"))

def train():
    frozen=np.load(OUT/"local-search-weights.npz")["weights"]
    rng=np.random.default_rng(981)
    mean=np.r_[frozen,.48,4.]
    spread=np.array([1.3,.5,.5,.055,1.,20.,.25,1.5])
    low=[3,1.1,.4,.03,2,15,.05,2.]
    high=[15,4,5,.42,8,180,1.,12.]
    finalists=[mean.copy()]
    for k in range(4):
        candidates=np.clip(rng.normal(mean,spread,(10,8)),low,high)
        candidates[0]=mean
        seeds=np.arange(1842000+100*k,1842040+100*k)
        ids=np.tile(seeds,len(candidates))
        a=run(ids,Transit(len(ids),np.repeat(candidates,len(seeds),axis=0)),Flight).reshape(len(candidates),len(seeds),4)
        elite=candidates[np.argsort((a[:,:,3]-.004*a[:,:,2]).mean(1))[-3:]]
        mean=elite.mean(0)
        spread=np.maximum(elite.std(0),spread*.5)
        finalists.append(mean.copy())
    scores=[]
    ids=np.arange(1850000,1850200)
    for weights in finalists:
        a=run(ids,Transit(len(ids),weights),Flight)
        scores.append((a[:,3]-.004*a[:,2]).mean())
    np.save(OUT/"transit-weights.npy",finalists[np.argmax(scores)])
    print(scores,flush=True)

if __name__=="__main__":
    train()
