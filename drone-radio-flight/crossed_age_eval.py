"""Cross every scenario with every link regime, with all checkpoints fixed before test."""
import argparse
import csv
from pathlib import Path
import numpy as np
from bullet_flight import BulletFlight
from flight import Flight
from mission import Worlds
from experiment import RADIOS

OUT = Path(__file__).resolve().parent / "results"
IDS = np.arange(3180000,3180120)
FIELDS = ("id","radio","policy","train_seed","found","crashed","capped_seconds","sent_bytes","received_bytes","buffer_evictions")

def evaluate(arm, backend):
    if arm=="transit":
        from transit_variant import make as transit
        build=lambda n,seed: transit(n)
    else:
        from age_experiment_policy import make
        def build(n,seed):
            policy=make(n,"learned" if arm=="masked" else arm,seed)
            policy.mask_reports=(arm=="masked")
            return policy
    seeds=range(5) if backend=="fast" and arm!="transit" else (0,)
    engine=Flight if backend=="fast" else BulletFlight
    path=OUT/f"crossed-{backend}-{arm}.csv"
    partial=path.with_suffix(".tmp")
    with partial.open("w",newline="") as handle:
        out=csv.writer(handle); out.writerow(FIELDS)
        for seed in seeds:
            for radio in RADIOS:
                for ids in np.array_split(IDS,1 if backend=="fast" else 3):
                    w=Worlds(ids,[radio]*len(ids),engine,event_radio=True)
                    try:
                        policy=build(len(ids),seed)
                        while w.active.any():
                            w.step(policy(w))
                        for i,scenario in enumerate(ids):
                            out.writerow((scenario,radio,arm,seed,int(w.found[i]),int(w.crashed[i]),.5*w.time[i],w.sent[i],w.received[i],w.evicted_unseen[i].sum()))
                    finally:
                        if backend=="bullet": w.flight.close()
            print(backend,arm,seed,flush=True)
    partial.replace(path)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=["transit","blind","fixed","learned","masked"])
    parser.add_argument("backend",choices=["fast","bullet"])
    args=parser.parse_args()
    evaluate(args.arm,args.backend)

