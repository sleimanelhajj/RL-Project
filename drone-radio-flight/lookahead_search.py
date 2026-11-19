"""Receding-horizon search with a proxy for information along each candidate leg.

Inspired by Hollinger and Sukhatme, RSS 2013 (doi:10.15607/RSS.2013.IX.051).
This is a two-sample path utility with episodic policy search, not their RIG planner.
"""
import numpy as np
from flight import Flight
from mission import CENTERS, GRID
from optimize_sweep import OUT, run
from transit_variant import Transit

class PathSearch(Transit):
    def __init__(self,n,weights):
        weights=np.broadcast_to(weights,(n,11))
        super().__init__(n,weights[:,:8])
        self.midpoint_weight=weights[:,8,None,None]
        self.neighbor_weight=weights[:,9,None,None]
        self.time_weight=weights[:,10,None,None]

    def waypoint_score(self,w,log_mass,distance,travel):
        mass=np.exp(log_mass).reshape(w.n,3,GRID,GRID)
        # The scan footprint overlaps adjacent three-meter grid cells.
        pad=np.pad(mass,((0,0),(0,0),(1,1),(1,1)))
        neighbors=pad[:,:,1:-1,:-2]+pad[:,:,1:-1,2:]+pad[:,:,:-2,1:-1]+pad[:,:,2:,1:-1]
        gain=(mass.reshape(log_mass.shape)+self.neighbor_weight*neighbors.reshape(log_mass.shape))
        midpoint=(w.flight.pos[:,:,:2][:,:,None,:]+CENTERS)/2
        intermediate=np.take_along_axis(gain,w.cell(midpoint),axis=-1)
        # Closely spaced scans have overlapping footprints. Discount the intermediate scan.
        separation=np.clip((distance-3)/6,0,1)
        path_gain=gain+self.midpoint_weight*intermediate*separation
        return np.log(path_gain+1e-20)-travel*distance-self.time_weight*np.log1p(distance/3)

def make(n):
    return PathSearch(n,np.load(OUT/"lookahead-search.npy"))

def train():
    rng=np.random.default_rng(872)
    frozen=np.load(OUT/"transit-weights.npy")
    mean=np.r_[frozen,.6,.45,.35]
    spread=np.array([1,.35,.4,.05,.7,15,.15,.65,.35,.25,.2])
    low=[4,1,.5,.025,2.5,30,.05,2,.1,.1,.02]
    high=[14,5,5,.43,8,200,1,8,2,1.5,1.4]
    finalists=[mean]
    for k in range(3):
        cohort=np.clip(rng.normal(mean,spread,(8,11)),low,high)
        cohort[0]=mean
        seeds=np.arange(2381000+100*k,2381032+100*k)
        ids=np.tile(seeds,len(cohort))
        a=run(ids,PathSearch(len(ids),np.repeat(cohort,len(seeds),axis=0)),Flight).reshape(len(cohort),len(seeds),4)
        elite=cohort[np.argsort((a[:,:,3]-.004*a[:,:,2]).mean(1))[-3:]]
        mean=elite.mean(0)
        spread=np.maximum(elite.std(0),spread*.55)
        finalists.append(mean)
        print("generation",k+1,flush=True)
    ids=np.arange(2420000,2420200)
    scores=[]
    for p in finalists:
        a=run(ids,PathSearch(len(ids),p),Flight)
        scores.append((a[:,3]-.004*a[:,2]).mean())
    np.save(OUT/"lookahead-search.npy",finalists[np.argmax(scores)])
    print("validation scores",scores,flush=True)
if __name__=="__main__":
    train()

