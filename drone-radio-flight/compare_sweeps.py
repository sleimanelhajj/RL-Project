"""Plot held-out results and report paired discordances with the strongest heuristic."""
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from optimize_sweep import OUT

scores = defaultdict(list)
with (OUT/"sweep-optimization.csv").open() as f:
    for r in csv.DictReader(f):
        scores[r["policy"]].append([float(r[k]) for k in ("found","crash","time_steps")])
scores["Contextual bandit"] = np.loadtxt(OUT/"bandit-holdout.csv",delimiter=",",skiprows=1)[:,1:4]
scores = {k:np.array(v) for k,v in scores.items()}
ref = scores["Default report sweep"][:,0]
for k,v in scores.items():
    print(k,v.mean(0),"wins/losses vs default",sum(v[:,0]>ref),sum(v[:,0]<ref))
fig,axs=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
for ax,j,title in zip(axs,(0,1),("Target found (%)","Mission crashed (%)")):
    values=[v[:,j].mean()*100 for v in scores.values()]
    bars=ax.barh(list(scores),values,color=["#39768a" if k!="Contextual bandit" else "#df8238" for k in scores])
    ax.bar_label(bars,fmt="%.0f"); ax.set(xlim=(0,108),title=title)
axs[0].invert_yaxis()
fig.suptitle("Drone search performance on 50 matched PyBullet missions")
fig.tight_layout()
fig.savefig(OUT/"pybullet-policy-comparison.png",dpi=170)
