"""Analyze learning-seed sensitivity and paired scenario-level outage effects."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from crossed_age_eval import OUT, IDS, RADIOS

def read(backend,arm):
    path=OUT/f"crossed-{backend}-{arm}.csv"
    rows=list(csv.DictReader(path.open()))
    seeds=sorted({int(r["train_seed"]) for r in rows})
    values=np.full((len(seeds),len(RADIOS),len(IDS),6),np.nan)
    for r in rows:
        key=(seeds.index(int(r["train_seed"])),RADIOS.index(r["radio"]),int(r["id"])-IDS[0])
        assert np.isnan(values[key][0])
        values[key]=[float(r[k]) for k in ("capped_seconds","found","crashed","sent_bytes","received_bytes","buffer_evictions")]
    assert np.isfinite(values).all()
    return values

def analyze():
    rng=np.random.default_rng(782)
    indices=rng.integers(0,len(IDS),(10000,len(IDS)))
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for ax,backend in zip(axes,("fast","bullet")):
        arms=["blind","fixed","learned"]
        if backend=="bullet": arms.append("transit")
        values={arm:read(backend,arm) for arm in arms}
        for arm,v in values.items():
            print(backend,arm,"averages (sec, found, crash, sent, received, evicted)",v.mean((0,1,2)).round(4))
            ax.plot(RADIOS,v[:,:,:,0].mean((0,2)),marker="o",label=arm)
        gain=values["fixed"][:,:,:,0]-values["learned"][:,:,:,0]
        diff=gain[:,3]-gain[:,1]
        interval=np.quantile(diff.mean(0)[indices].mean(1),[.025,.975])
        print(backend,"extra seconds saved in burst25 vs scattered",diff.mean(),"mission CI",interval,"per seed",diff.mean(1))
        print(backend,"mean gain by radio",gain.mean((0,2)))
        ax.set(title="Fast simulator, five training seeds" if backend=="fast" else "PyBullet, prespecified training seed 0",ylabel="Mean capped search time (s)")
        ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"crossed-age-comparison.png",dpi=175)

if __name__=="__main__": analyze()
