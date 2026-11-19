"""Select a coordinated sweep by episodic return, then benchmark on held-out Bullet missions."""
import csv
from pathlib import Path
import numpy as np
import torch
from bullet_flight import BulletFlight
from experiment import Actor, RADIOS
from flight import Flight
from mission import Worlds, Sweep, greedy, steer

OUT = Path(__file__).resolve().parent / "results"

class ReportSweep:
    def __init__(self, n, spacing, depth):
        self.index = np.zeros((n, 3), int)
        # Three parallel strips, each with two sampling columns.
        self.spacing = spacing
        self.depth = depth

    def __call__(self, w):
        s = self.spacing
        row = (self.index // 2) % 8
        col = np.where(row % 2 == 0, self.index % 2, 1 - self.index % 2)
        center = w.forecast()
        x = np.clip(center[:, 0, None], 2 + 2.5*s, 40 - 2.5*s)
        x = x + (2*np.arange(3) + col - 2.5)*s
        y = np.clip(center[:, 1, None] - self.depth + row*s, 2, 40)
        action = steer(w.flight.pos[..., :2], np.stack([x, y], -1))
        self.index += action == 8
        return action

def run(ids, policy, engine):
    w = Worlds(ids, [RADIOS[i % 5] for i in ids], flight_factory=engine)
    try:
        while w.active.any():
            w.step(policy(w))
        return np.stack([w.found, w.crashed, w.time, w.returns], axis=1)
    finally:
        if engine is BulletFlight:
            w.flight.close()

def main():
    torch.set_num_threads(1)
    train = np.arange(321000, 321100)
    choices = [(s,d) for s in (3.5,4.5,5.5) for d in (6,10,14)]
    scores = [run(train, ReportSweep(len(train), *p), Flight)[:,3].mean() for p in choices]
    chosen = choices[np.argmax(scores)]
    print("Training returns", list(zip(choices,scores)), "selected",chosen, flush=True)
    ids = np.arange(432000,432050)
    actor = Actor(False,False)
    actor.load_state_dict(torch.load(OUT / "policy-0-101.pt",map_location="cpu",weights_only=True))
    @torch.no_grad()
    def learned(w):
        return actor(torch.from_numpy(w.features())[None])[0][0].argmax(-1).numpy()
    methods = {"Tuned sweep": ReportSweep(len(ids),*chosen), "Default report sweep":ReportSweep(len(ids),4.5,10),
               "Sector sweep":Sweep(len(ids)), "Sector greedy":lambda w:greedy(w,False),
               "Radio-aware greedy":lambda w:greedy(w,True), "FF MAPPO (101)":learned}
    import matplotlib.pyplot as plt
    totals = []
    with (OUT/"sweep-optimization.csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy","seed","radio","found","crash","time_steps","return"])
        for name,policy in methods.items():
            a = run(ids,policy,BulletFlight)
            for i,r in zip(ids,a):
                writer.writerow([name,i,RADIOS[i%5],*r])
            f.flush()
            totals.append(a.mean(0))
            print(name,a.sum(0),flush=True)
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,j,title in zip(axs,(0,1),("Target found (%)","Mission crashed (%)")):
        values=100*np.array(totals)[:,j]
        bars=ax.barh(list(methods),values,color=["#007b83"]+["#778ca3"]*5)
        ax.bar_label(bars,fmt="%.0f"); ax.set(xlim=(0,108),title=title)
    axs[0].invert_yaxis()
    fig.suptitle("50 matched PyBullet missions across five radio conditions")
    fig.tight_layout()
    fig.savefig(OUT/"sweep-optimization.png",dpi=160)

if __name__ == "__main__":
    main()

