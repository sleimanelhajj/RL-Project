# RL Project: Multi-drone search in PyBullet

This repository contains a from-scratch reinforcement-learning simulation for cooperative drone search with a hidden moving target and unreliable radio communication. The active code is in [`drone-radio-flight/`](drone-radio-flight/).

The project includes a Gymnasium-style vectorized mission loop, a fast flight backend, PyBullet rigid-body validation, hand-coded baselines, MAPPO checkpoints, and learned policy-search refinements. See [`FUTURE_AGENTS.md`](FUTURE_AGENTS.md) for the exact current result, reproduction commands, split policy, and planned experiments.

## Latest simulation result

The transit-scanning policy (`drone-radio-flight/transit_variant.py`) found 287/300 targets with zero crashes on fresh matched PyBullet missions. The graph is [`drone-radio-flight/results/refinement-300.png`](drone-radio-flight/results/refinement-300.png).

## Scope

This is a simulation research project. Results should be replicated across independent training seeds and larger held-out sets before being treated as a general algorithmic claim.
