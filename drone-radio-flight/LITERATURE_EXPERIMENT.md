# Literature-based search experiment

Three drones search for one moving target. The decentralized policy uses its local scan history and delivered teammate reports. Both new policies use episodic cross-entropy policy search over engineered decision rules. The MAPPO checkpoints belong to an earlier experiment.

## Methods and motivation

* **Lookahead search** scores scan opportunities at the proposed waypoint and at a midpoint along the path, discounts overlap, and penalizes travel. This approximates informative path planning under a travel budget. [Hollinger and Sukhatme, RSS 2013](https://roboticsproceedings.org/rss09/p51.html)
* **Miss-likelihood search** downweights locations after unsuccessful scans and scores remaining probability across the sensor footprint. It uses an approximate local probability map and timestamped radio evidence. [Moving-target UAV search, Sensors 2021](https://www.mdpi.com/1424-8220/21/19/6568)

## Selection and results

The shared selection metric was mean episodic return minus 0.004 times capped mission steps. Fast-simulator validation seeds 2420000 through 2420199 selected lookahead (0.79815), followed by miss-likelihood (0.79600) and frozen transit (0.77205). All four policies were evaluated on identical PyBullet seeds 2520000 through 2520299, 60 per radio condition.

| Policy | Detections / 300 | Mean time, misses capped at 50 s | Crashes |
|---|---:|---:|---:|
| Lookahead | 286 | 14.49 s | 0 |
| Miss-likelihood | 289 | 14.82 s | 0 |
| Frozen transit | 284 | 14.56 s | 0 |
| Fixed scan | 274 | 19.64 s | 0 |

Lookahead saved 5.15 s vs fixed scan (paired, radio-stratified bootstrap 95% interval 3.58 to 6.71 s). Against transit, the detection and time intervals both included zero. The highest detection count belonged to the secondary candidate. The next independent test should use multiple training seeds, freeze a single primary endpoint, and estimate radio-specific gains by replaying the same scenarios under each radio condition. The present test assigns a single radio condition to each scenario.

## Reproduce

Use Python 3.13.7 and the dependency versions recorded in requirements.txt. All commands below run inside drone-radio-flight. The stored weights allow evaluation without retraining.

```powershell
python -m pip install -r requirements.txt
python literature_eval.py lookahead_search
python literature_eval.py belief_update
python literature_eval.py transit_variant
python literature_eval.py fixed
python plot_literature.py lookahead_search
```

To retrain, run lookahead_search.py and belief_update.py in a working copy. Those training scripts write their chosen weights into results. Per-mission output is in results/literature-*.csv. The replay GIF in this project was generated with an earlier policy.
