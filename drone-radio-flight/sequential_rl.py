"""Two-stage fitted Q-learning over coordinated search options."""
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from mission import HORIZON, Worlds
from experiment import RADIOS, Actor
from optimize_sweep import ReportSweep, OUT, run
from bullet_flight import BulletFlight
from flight import Flight

SWITCH = 40
FIRST = [(4.5,10,0), (4.5,10,3.8), (5,7,3.5)]
SECOND = [(0,0), (0,3.5), (3,3.5)]
def features(w):
    x,y = ((w.report-[21,22])/[10,8]).T
    return np.column_stack([np.ones(w.n),x,y,x*x,y*y,x*y,w.estimated_drift*6])

class Options:
    def __init__(self,n):
        self.route = ReportSweep(n,4.5,10)
        self.last = np.full((n,3,2),-100.)
    def start(self, idx):
        self.first = idx
        self.route.spacing,self.route.depth,self.interval = np.array(FIRST)[idx].T[:,:,None]
    def state(self,w):
        return np.column_stack([features(w),np.eye(len(FIRST))[self.first],
            self.route.index/12,w.team.mean(axis=1),w.forecast()/42])
    def switch(self,idx):
        shift,period = np.array(SECOND)[idx].T
        self.route.index += shift.astype(int)[:,None]
        self.interval = period[:,None]
    def __call__(self,w):
        action = self.route(w)
        xy = w.flight.pos[...,:2]
        extra = ((self.interval>0) & (np.linalg.norm(xy-self.last,axis=-1)>self.interval)
                 & (np.linalg.norm(xy-w.forecast()[:,None],axis=-1)<17))
        action[extra]=8
        self.last[action==8]=xy[action==8]
        return action

def fit(x,y):
    reg = np.eye(x.shape[1])*5
    reg[0,0]=0
    return np.linalg.solve(x.T@x+reg,x.T@y)

def train():
    ids=np.arange(672000,672450)
    states=[]; tails=[]; early=[]
    for first in range(len(FIRST)):
        branch=[]
        for second in range(len(SECOND)):
            w=Worlds(ids,[RADIOS[i%5] for i in ids])
            p=Options(w.n); p.start(np.full(w.n,first))
            for _ in range(SWITCH):
                w.step(p(w))
            if second==0:
                states.append(p.state(w)); early.append((w.returns.copy(),w.active.copy()))
            p.switch(np.full(w.n,second))
            prefix=w.returns.copy()
            while w.active.any():
                w.step(p(w))
            branch.append(w.returns-prefix)
        tails.append(np.column_stack(branch))
    # Terminal missions contribute their prefix return only.
    active=np.concatenate([a for _,a in early])
    q2=fit(np.vstack(states)[active],np.vstack(tails)[active])
    targets=np.column_stack([r+a*(s@q2).max(1) for s,(r,a) in zip(states,early)])
    return fit(features(w),targets),q2

def learned(n,q):
    p=Options(n)
    def act(w):
        if w.t==0: p.start((features(w)@q[0]).argmax(1))
        if w.t==SWITCH: p.switch((p.state(w)@q[1]).argmax(1))
        return p(w)
    return act

def fixed(n):
    p=Options(n)
    p.start(np.ones(n,int))
    def act(w):
        if w.t==SWITCH: p.switch(np.ones(n,int))
        return p(w)
    return act

if __name__=="__main__":
    torch.set_num_threads(1)
    q=train()
    np.savez(OUT/"search-option-q.npz",q1=q[0],q2=q[1])
    actor=Actor(False,False)
    actor.load_state_dict(torch.load(OUT/"policy-0-101.pt",weights_only=True))
    @torch.no_grad()
    def mappo(w):
        return actor(torch.from_numpy(w.features())[None])[0][0].argmax(-1).numpy()
    methods={"Fitted Q-learning":lambda n:learned(n,q),"Fixed scanning option":fixed,
             "Report-centered sweep":lambda n:ReportSweep(n,4.5,10),"Original MAPPO":lambda n:mappo}
    ids=np.arange(778000,778100)
    results={}
    with (OUT/"sequential-rl.csv").open("w",newline="") as f:
        out=csv.writer(f); out.writerow(["policy","seed","radio","found","crash","steps_to_find_capped","return"])
        for name,make in methods.items():
            a=np.vstack([run(group,make(len(group)),BulletFlight) for group in np.array_split(ids,2)])
            results[name]=a
            for i,row in zip(ids,a): out.writerow([name,i,RADIOS[i%5],*row])
            f.flush()
            print(name,a.mean(0),flush=True)
    fig,ax=plt.subplots(figsize=(8,4.6))
    for name,a in results.items():
        t=np.arange(HORIZON+1)
        ax.step(t*.5,100*((a[:,0,None]>0)&(a[:,2,None]<=t)).mean(0),where="post",label=name)
    ax.set(xlabel="Simulated mission time (seconds)",ylabel="Missions with a detection (%)",ylim=(0,102),xlim=(0,50),title="Search performance on 100 matched PyBullet missions")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"sequential-rl.png",dpi=180)

