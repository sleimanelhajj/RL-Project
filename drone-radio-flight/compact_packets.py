"""Closed-loop comparison of two-record packets under a common slot budget."""
import csv
from pathlib import Path
import numpy as np
from mission import CENTERS, DRONES, HORIZON, LINKS, Worlds
from flight import Flight
from age_experiment_policy import make

OUT=Path(__file__).resolve().parent/"results"

class CompactWorlds(Worlds):
    def __init__(self,ids,radios,flight_factory=Flight,*,packet_policy="fifo",novelty_weight=2.):
        super().__init__(ids,radios,flight_factory)
        if packet_policy not in ("fifo","novelty"):
            raise ValueError(packet_policy)
        self.packet_policy=packet_policy
        self.novelty_weight=novelty_weight
        self.seen=np.zeros((self.n,DRONES,DRONES,HORIZON+1),bool)
        self.evicted_unseen=np.zeros((self.n,DRONES,DRONES),int)

    def allow_packet(self, src, dst, ticks):
        return True

    def step(self,actions):
        active=self.active.copy()
        scans=(actions==8)&active[:,None]
        rows,src=np.nonzero(scans)
        old=self.last_stamp[rows,src,self.last_slot[rows,src]]
        for dst in range(DRONES):
            lost=(old>0)&(src!=dst)&~self.seen[rows,dst,src,old]
            self.evicted_unseen[rows,src,dst]+=lost
        sent=self.sent.copy()
        received=self.received.copy()
        remote=self.remote.copy()
        result=super().step(actions)
        # Replace the superclass's full-ring channel before the next policy observation.
        self.sent,self.received,self.remote=sent,received,remote
        events=[]
        for link,(src,dst) in enumerate(LINKS):
            ticks=self.last_stamp[:,src]
            cells=self.last_cells[:,src]
            recent=ticks.argmax(1)
            score=ticks.astype(float)
            if self.packet_policy=="novelty":
                distances=np.linalg.norm(CENTERS[cells]-CENTERS[cells[self.world,recent]][:,None],axis=-1)
                score+=self.novelty_weight*np.minimum(distances,12)
            score[ticks==0]=-np.inf
            score[self.world,recent]=-np.inf
            second=score.argmax(1)
            allowed=self.allow_packet(src,dst,ticks)
            for idx,valid in [(recent,ticks.max(1)>0),(second,np.isfinite(score.max(1)))]:
                send=active&self.transmit&valid&allowed
                deliver=send&~self.trace[self.t-1,:,link]
                cost=8 if idx is recent else 4
                self.sent+=cost*send
                self.received+=cost*deliver
                r=np.flatnonzero(deliver)
                tick=ticks[r,idx[r]]
                cell=cells[r,idx[r]]
                mask=~self.seen[r,dst,src,tick]
                r,tick,cell=r[mask],tick[mask],cell[mask]
                if len(r):
                    self.seen[r,dst,src,tick]=True
                    np.maximum.at(self.remote[:,dst],(r,cell),tick)
                    events.append((r,np.full_like(r,dst),np.full_like(r,src),cell,tick))
        self.new_scan_events=(tuple(np.concatenate(v) for v in zip(*events))
                              if events else tuple(np.empty(0,int) for _ in range(5)))
        return result

def evaluate(ids,channels,weight):
    w=CompactWorlds(ids,channels,packet_policy="novelty" if weight else "fifo",novelty_weight=weight)
    p=make(len(ids),"fixed",0)
    while w.active.any(): w.step(p(w))
    return np.column_stack([w.time*.5,w.found,w.crashed,w.sent,w.received,w.evicted_unseen.sum((1,2)),w.returns])

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--start",type=int,default=4050000)
    parser.add_argument("--count",type=int,default=120)
    args=parser.parse_args()
    ids=np.repeat(np.arange(args.start,args.start+args.count),2)
    channels=np.tile(["burst25","scattered"],args.count)
    with (OUT/f"compact-validation-{args.start}.csv").open("w",newline="") as f:
        out=csv.writer(f)
        out.writerow(["id","radio","novelty_weight","seconds","found","crashed","sent","received","evictions","return"])
        for weight in [0,.5,2,5]:
            a=evaluate(ids,channels,weight)
            for i,c,row in zip(ids,channels,a): out.writerow([i,c,weight,*row])
            print(weight,a.mean(0),flush=True)

