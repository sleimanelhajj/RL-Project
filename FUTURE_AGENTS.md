# Future-agent handoff

**Latest experiment:** [Literature-based search policies, results, and reproduction commands](drone-radio-flight/LITERATURE_EXPERIMENT.md).

## Project

`drone-radio-flight` is a three-drone search-and-rescue simulation. It contains a lightweight vectorized flight model (`flight.py`), an optional PyBullet rigid-body backend (`bullet_flight.py`), mission and radio model (`mission.py`), MAPPO training/evaluation (`experiment.py`), and policy-search refinement scripts.

The older `drone-marl-sar` directory from the workspace is a separate pilot/prototype. It is not the source of the results documented here. Future agents should start with `drone-radio-flight` and consult the older pilot only when deliberately comparing designs.

## Previous experiment

The reference controller is the transit-scanning policy in `transit_variant.py`. It uses a learned waypoint score and predicts where the drone will be when it reaches a scan point. It can also scan during transit when it has moved sufficiently far from its last scan and is near the target forecast.

On 300 fresh matched PyBullet missions (`results/optimization-transit_variant.csv`), it found 287/300 targets with 0 crashes and mean capped time 14.16 s. The previous local-search RL controller found 281/300 with 0 crashes and mean capped time 14.85 s. The fixed scanning maneuver found 283/300 with 0 crashes and mean capped time 17.85 s. `results/refinement-300.png` visualizes the comparison.

These results are simulation results, not evidence of real-drone safety. The 300-mission holdout was used only after tuning and validation. Replicate with independent training seeds before claiming a general improvement over the prior RL policy.

## Reproduction

From the drone-radio-flight directory, use the project virtual environment supplied by the user or install `numpy`, `torch`, `matplotlib`, and `pybullet`.

```powershell
python optimize_sweep.py
python local_search_rl.py
python transit_variant.py
python evaluate_refinement.py transit_variant
python plot_refinement.py transit_variant
```

The scripts use deterministic seed ranges for training, validation, and holdout evaluation. Do not mix holdout IDs into tuning.

## Important design notes

- The environment has 3 drones, 100 half-second steps, a 42 by 42 search area, a moving hidden target, detector dropouts, and five radio conditions.
- The reward is `-0.001` per active step, `+1` for detection, and `-1` for a crash.
- `Worlds` is the environment. `Flight` is the fast vectorized backend and `BulletFlight` is the PyBullet validation backend.
- The current policy uses team-visible scan state plus received radio evidence. For a strict decentralized study, ensure every feature is restricted to information actually received by each drone.
- Keep `results/*.csv` with each reported graph. The `*.pt`, `*.npz`, and `*.npy` files are learned checkpoints/weights.

## Next research steps

1. Run at least 5 independent training seeds for `transit_variant.py`.
2. Evaluate all seeds on one untouched 1,000-mission PyBullet holdout with confidence intervals.
3. Add partial-observability tests and radio ablations. Report reliable, scattered, burst5, burst25, and off separately.
4. Compare PPO/MAPPO, recurrent MAPPO, SAC or TD3 where the action space is made continuous, and the fixed maneuver under identical seeds.
5. Test robustness to wind, target drift, sensor range, radio-loss rate, and obstacle layouts.

