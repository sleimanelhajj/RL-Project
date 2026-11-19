"""Compare first detection on matched mission seeds, counting misses at the deadline."""
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from optimize_sweep import OUT

rows=defaultdict(list)
with (OUT/"sequential-rl.csv").open() as handle:
    for row in csv.DictReader(handle):
        rows[row["policy"]].append([float(row[k]) for k in ("seed","found","crash","steps_to_find_capped","return")])
rows["MAPPO + 40 updates"]=np.loadtxt(OUT/"continued-mappo"/"holdout.csv",delimiter=",",skiprows=1)
rows={k:np.array(v) for k,v in rows.items()}
order=["Original MAPPO","MAPPO + 40 updates","Report-centered sweep","Fixed scanning option","Fitted Q-learning"]
fig,ax=plt.subplots(figsize=(8.3,4.8))
for name,color in zip(order,["#afb4ba","#6a4c93","#435665","#e8883b","#007c76"]):
    data=rows[name]
    assert np.array_equal(data[:,0],np.arange(778000,778100))
    t=np.arange(101)
    ax.step(t*.5,100*((data[:,1,None]>0)&(data[:,3,None]<=t)).mean(0),where="post",label=name,color=color,lw=2)
    print(name, data[:,1:].mean(0),flush=True)
for a,b in [(order[1],order[0]),(order[4],order[2]),(order[4],order[3])]:
    d=rows[a][:,1]-rows[b][:,1]
    rng=np.random.default_rng(2718)
    print(a,"vs",b,"detection pp CI",np.quantile(rng.choice(d,(12000,len(d))).mean(1)*100,[.025,.975]))
ax.set(xlabel="Simulated mission time (seconds)",ylabel="Missions with a detection (%)",ylim=(0,102),xlim=(0,50),title="When did the drones find the target?")
ax.legend(title="100 matched PyBullet missions",loc="lower right"); ax.grid(alpha=.18)
fig.tight_layout(); fig.savefig(OUT/"rl-training-comparison.png",dpi=180)
