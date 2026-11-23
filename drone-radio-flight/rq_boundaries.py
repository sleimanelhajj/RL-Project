"""Frozen-policy boundary probes. Shared detector draws and blocked-slot counts per paired scenario."""
import csv
import numpy as np
from sender_schedule import Scheduled, OUT
from age_experiment_policy import make
from flight import Flight
from mission import HORIZON, LINKS

if __name__=="__main__":
    ids=np.arange(5350000,5350160)
    with (OUT/"rq-boundaries.csv").open("w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["condition","arm","seed","found","seconds","bytes"])
        for name,pd,blocked in [("perfect_link",.95,0),("weak_sensor_short",.4,5),("weak_sensor_long",.4,50),("strong_sensor_long",.95,50),("blackout",.4,100)]:
            for arm,params in [("once",(1,101)),("trained",(2,8)),("always",(1,1))]:
                w=Scheduled(ids,["reliable"]*len(ids),Flight,params)
                w.detectors*=.75/pd
                for i,seed in enumerate(ids):
                    offsets=np.random.default_rng([int(seed),711]).integers(0,HORIZON,len(LINKS))
                    # 50% loss, separated five-slot bursts or one fifty-slot burst
                    for link,start in enumerate(offsets):
                        slots=np.arange(blocked) if blocked!=5 else np.arange(100)[np.arange(100)%10<5]
                        w.trace[(slots+start)%100,i,link]=True
                policy=make(w.n,"fixed",0)
                while w.active.any(): w.step(policy(w))
                for i,s in enumerate(ids): writer.writerow([name,arm,s,int(w.found[i]),w.time[i]*.5,w.sent[i]])
                print(name,arm,w.found.sum(),w.time.mean()*.5,w.sent.mean(),flush=True)
