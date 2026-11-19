"""Matched, held-out evaluation of decentralized search policies."""
import sys
import numpy as np
from optimize_sweep import run, OUT
from bullet_flight import BulletFlight
from sequential_rl import fixed, learned

def evaluate(name, make):
    ids=np.arange(1450000,1450300)
    parts=[]
    for group in np.array_split(ids,10):
        parts.append(run(group,make(len(group)),BulletFlight))
    a=np.vstack(parts)
    np.savetxt(OUT/("confirm-"+name+".csv"),np.column_stack([ids,a]),delimiter=",",header="seed,found,crash,capped_steps,return",comments="")
    print(name,a.mean(0),flush=True)

if __name__=="__main__":
    if sys.argv[1]=="fixed":
        evaluate("fixed",fixed)
    else:
        weights=np.load(OUT/"search-option-q.npz")
        evaluate("previous",lambda n:learned(n,(weights["q1"],weights["q2"])))
