"""Frozen-sender negative controls on fresh, crossed fast-simulator missions."""
import csv
import numpy as np
from age_experiment_policy import OUT
from sender_schedule import play

def main():
    ids=np.arange(5120000,5120160)
    radios=("reliable","scattered","burst25","off")
    results={}
    # In the mission generator the radio seed is independent of the scenario seed.
    silent=play(ids,["off"]*len(ids),(1,101))
    with (OUT/"sender-falsification.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("scenario","radio","policy","found","crash","capped_seconds","sent","received","return"))
        for mode,params in (("silent",None),("event",(1,101)),("retry",np.load(OUT/"sender-schedule.npy"))):
            totals=[]
            for radio in radios:
                a=silent if mode=="silent" else play(ids,[radio]*len(ids),params)
                totals.append(a)
                for key,row in zip(ids,a):
                    writer.writerow((key,radio,mode,*row))
            results[mode]=np.stack(totals)
            print(mode,results[mode][:,:,[0,2,3]].mean(1),flush=True)
    for mode,v in results.items():
        np.testing.assert_array_equal(v[-1],silent)
    rng=np.random.default_rng(700)
    i=rng.integers(0,len(ids),(10000,len(ids)))
    for other in ("silent","event"):
        for name,d in zip(radios,results["retry"]-results[other]):
            print(other,name,"retry time difference and CI",d[:,2].mean(),np.quantile(d[i,2].mean(1),[.025,.975]))
        change=(results["retry"]-results[other])[2,:,2]-(results["retry"]-results[other])[1,:,2]
        print(other,"burst interaction",change.mean(),np.quantile(change[i].mean(1),[.025,.975]))
if __name__=="__main__": main()
