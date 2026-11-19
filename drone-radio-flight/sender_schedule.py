"""Budget-aware direct policy search over open-loop retransmission schedules."""
import csv
from pathlib import Path
import sys
import numpy as np
from compact_packets import CompactWorlds
from age_experiment_policy import make,OUT
from mission import Worlds
from flight import Flight
from bullet_flight import BulletFlight

class Scheduled(CompactWorlds):
    def __init__(self,ids,radios,engine,params):
        super().__init__(ids,radios,engine,packet_policy="novelty")
        self.repeats,self.period=params
    def allow_packet(self,src,dst,ticks):
        age=self.t-ticks.max(1)
        return (age<self.repeats)|(age%self.period==0)

def play(ids,radios,params,engine=Flight):
    if params is None:
        w=Worlds(ids,radios,engine,event_radio=True)
    else:
        w=Scheduled(ids,radios,engine,params)
    try:
        p=make(w.n,"fixed",0)
        while w.active.any():
            w.step(p(w))
        return np.column_stack([w.found,w.crashed,w.time*.5,w.sent,w.received,w.returns])
    finally:
        if engine is BulletFlight: w.flight.close()

def scenarios(start,count):
    return np.repeat(np.arange(start,start+count),2),np.tile(["scattered","burst25"],count)

def train():
    # 1,000 transmitted bytes cost the same as 1 s of capped search time.
    options=[(r,p) for r in (1,2,3) for p in (3,8,16)]
    ids,radios=scenarios(4830000,60)
    scores=[]
    for params in options:
        a=play(ids,radios,params)
        scores.append((a[:,5]-.008*a[:,2]-.000008*a[:,3]).mean())
    chosen=options[np.argmax(scores)]
    np.save(OUT/"sender-schedule.npy",chosen)
    np.savetxt(OUT/"sender-training.csv",np.column_stack([options,scores]),delimiter=",",header="repeat,period,utility",comments="")
    print(chosen,flush=True)

if __name__=="__main__":
    name=sys.argv[1]
    if name=="train": train()
    else:
        ids,radios=scenarios(4910000,140)
        params={"full":None,"compact":(1,1),"once":(1,101),"learned":tuple(np.load(OUT/"sender-schedule.npy")) if name=="learned" else None}[name]
        blocks=np.array_split(np.arange(len(ids)),7)
        a=np.vstack([play(ids[b],radios[b],params,BulletFlight) for b in blocks])
        np.savetxt(OUT/f"sender-bullet-{name}.csv",np.column_stack([ids,np.arange(len(ids))%2,a]),delimiter=",",header="seed,radio_index,found,crashed,capped_seconds,sent,received,return",comments="")
        print(name,a.mean(0))
