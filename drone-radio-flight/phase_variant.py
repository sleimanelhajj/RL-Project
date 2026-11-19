"""Learn separate early and late waypoint scores with episodic policy search."""
import numpy as np
from local_search_rl import LocalSearch, LOW,HIGH
from optimize_sweep import OUT, run
from flight import Flight

class Adaptive(LocalSearch):
    def __init__(self,n,params):
        self.params=np.broadcast_to(params,(n,12))
        super().__init__(n,self.params[:,:6])
    def __call__(self,w):
        if w.t==32:
            self.weights=self.params[:,6:].T[:,:,None,None]
        return super().__call__(w)

def make(n):
    return Adaptive(n,np.load(OUT/"phase-refinement.npy"))

if __name__=="__main__":
    base=np.tile(np.load(OUT/"local-search-weights.npz")["weights"],2)
    rng=np.random.default_rng(744)
    center=base.copy()
    spread=np.tile(HIGH-LOW,2)*.16
    history=[base]
    for j in range(4):
        candidates=np.clip(rng.normal(center,spread,(8,12)),np.tile(LOW,2),np.tile(HIGH,2))
        candidates[0]=center
        seeds=np.arange(1710000+100*j,1710050+100*j)
        ids=np.tile(seeds,len(candidates))
        a=run(ids,Adaptive(len(ids),np.repeat(candidates,len(seeds),axis=0)),Flight).reshape(8,len(seeds),4)
        score=(a[:,:,3]-.004*a[:,:,2]).mean(1)
        elite=candidates[np.argsort(score)[-3:]]
        center=elite.mean(0); history.append(center)
        spread=np.maximum(elite.std(0),np.tile(HIGH-LOW,2)*.05)
    ids=np.arange(1850000,1850200)
    scores=[]
    for p in history:
        a=run(ids,Adaptive(len(ids),p),Flight)
        scores.append((a[:,3]-.004*a[:,2]).mean())
    np.save(OUT/"phase-refinement.npy",history[np.argmax(scores)])
    print("Validation scores",scores,flush=True)
