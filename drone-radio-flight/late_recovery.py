"""Search for a late-mission recovery policy using miss-weighted episodic returns."""
import numpy as np
from transit_variant import Transit, make as frozen
from optimize_sweep import OUT, run
from flight import Flight

class Recovery(Transit):
    def __init__(self,n,params):
        self.base=np.load(OUT/"transit-weights.npy")
        super().__init__(n,self.base)
        self.params=np.broadcast_to(params,(n,5))

    def __call__(self,w):
        onset,scale,width,travel,penalty=self.params.T
        phase=np.clip((w.t-onset)/22.,0.,1.)
        steps=np.stack([scale,np.zeros(w.n),penalty,travel,width,np.zeros(w.n)],axis=0)
        self.weights=(self.base[:6,None]*np.exp(steps*phase))[:,:,None,None]
        return super().__call__(w)

def objective(results):
    return (results[...,3]-.003*results[...,2]-.8*(1-results[...,0])).mean(-1)

def make(n):
    return Recovery(n,np.load(OUT/"late-recovery.npy"))

def main():
    rng=np.random.default_rng(6541)
    mean=np.array([32.,.4,.25,-.35,.4])
    spread=np.array([8.,.4,.35,.35,.35])
    low=[12,-.4,-.6,-1.3,-.4]
    high=[55,1.2,.9,.6,1.3]
    options=[mean.copy(),np.array([32.,0,0,0,0])]
    for g in range(3):
        p=np.clip(rng.normal(mean,spread,(7,5)),low,high)
        ids=np.arange(4610000+100*g,4610040+100*g)
        a=run(np.tile(ids,len(p)),Recovery(len(ids)*len(p),np.repeat(p,len(ids),axis=0)),Flight)
        elite=p[np.argsort(objective(a.reshape(len(p),len(ids),4)))[-3:]]
        mean=elite.mean(0)
        spread=np.maximum(elite.std(0),spread*.6)
        options.append(mean.copy())
    ids=np.arange(4620000,4620200)
    scores=[]
    for p in options:
        scores.append(objective(run(ids,Recovery(len(ids),p),Flight)))
    np.save(OUT/"late-recovery.npy",options[np.argmax(scores)])
    print("validation",scores,"chosen",np.argmax(scores),flush=True)
    ids=np.arange(4630000,4630300)
    old=run(ids,frozen(len(ids)),Flight)
    new=run(ids,make(len(ids)),Flight)
    np.savetxt(OUT/"late-recovery-holdout.csv",np.column_stack([ids,old,new]),delimiter=",",header="seed,baseline_found,baseline_crash,baseline_time_steps,baseline_return,new_found,new_crash,new_time_steps,new_return",comments="")
    diff=new-old
    indices=rng.integers(0,len(ids),(10000,len(ids)))
    print("holdout old/new",old.mean(0),new.mean(0),"paired 95% interval found,seconds",np.quantile(diff[indices,0].mean(1),[.025,.975]),np.quantile(diff[indices,2].mean(1)*.5,[.025,.975]),flush=True)
if __name__=="__main__":
    main()
