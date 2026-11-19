"""Full-feedback contextual bandit over nine team search maneuvers."""
import csv
import numpy as np
from bullet_flight import BulletFlight
from flight import Flight
from mission import Worlds
from optimize_sweep import OUT, ReportSweep, run

def context(w):
    x,y = ((w.report - [21,22]) / [10,8]).T
    return np.column_stack([np.ones(w.n), x, y, x*y, x*x, y*y])

if __name__ == "__main__":
    train = np.arange(321000,321400)
    choices = np.array([(s,d) for s in (3.5,4.5,5.5) for d in (6,10,14)])
    features = context(Worlds(train,["off"]*len(train)))
    rewards = np.column_stack([run(train,ReportSweep(len(train),*p),Flight)[:,3] for p in choices])
    weights = np.linalg.solve(features.T @ features + np.diag([0,12,12,12,12,12]),features.T @ rewards)
    ids = np.arange(432000,432050)
    def policy(w):
        if w.t == 0:
            spacing, depth = choices[(context(w) @ weights).argmax(1)].T
            policy.sweep = ReportSweep(w.n,spacing[:,None],depth[:,None])
        return policy.sweep(w)
    a = run(ids,policy,BulletFlight)
    print("Bandit totals",a.sum(0),flush=True)
    np.savetxt(OUT/"bandit-holdout.csv", np.column_stack([ids,a]),delimiter=",",header="seed,found,crash,time_steps,return",comments="")
