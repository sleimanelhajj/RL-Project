"""Continue the saved MAPPO actor with a fresh critic and evaluate on the same missions."""
import numpy as np
import torch
from experiment import Actor, experiment, output_csv, RADIOS
from optimize_sweep import OUT, run
from bullet_flight import BulletFlight
from flight import Flight

def policy(model):
    @torch.no_grad()
    def act(w):
        return model(torch.from_numpy(w.features())[None])[0][0].argmax(-1).numpy()
    return act

if __name__=="__main__":
    torch.set_num_threads(1)
    folder=OUT/"continued-mappo"; folder.mkdir(exist_ok=True)
    initial=torch.load(OUT/"policy-0-101.pt",weights_only=True)
    actor,curve=experiment("FF",7251,40,32,folder,initial_actor=initial)
    output_csv(folder/"curve.csv",curve)
    ids=np.arange(778000,778100)
    a=np.vstack([run(group,policy(actor),BulletFlight) for group in np.array_split(ids,2)])
    np.savetxt(folder/"holdout.csv",np.column_stack([ids,a]),delimiter=",",header="seed,found,crash,time_steps,return",comments="")
    print("Continued MAPPO",a.mean(0),flush=True)
