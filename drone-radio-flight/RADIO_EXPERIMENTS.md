# Event-aware drone search experiments

## What each experiment tests

* Event radio records a source and observation tick for each scan, retains the most recent eight source records, suppresses duplicate evidence from retransmission, and counts unseen records evicted during outages. The historical cell-map interface remains the default.
* The age trial freezes the transit navigation policy and trains teammate evidence strength and forgetting time by episodic policy search. Blind decay starts when a report arrives. Fixed decay starts when a scan occurred. Learned decay also depends on estimated drift speed. Five seeds shared matched training budgets. Fast simulator tests use all five checkpoints and PyBullet uses prespecified seed 0. The seed-0 fixed and learned checkpoints selected identical effective parameters. Uncertainty intervals from this test resample scenarios.
* The packet trial freezes navigation and receiver parameters. Compact packets always contain the newest scan and can include a second spatially distinct scan from the eight-record buffer. A two-most-recent-record control isolates spatial selection. Each transmission has a 4-byte header and 4 bytes per record. The simulator models slot loss; it does not model collisions between packets, variable loss by packet length, or radio power consumption.
* The sender trial optimizes a low-dimensional retry schedule over nine options. Training utility is episode return minus 0.008 times capped seconds minus 0.000008 times transmitted bytes. The sender sees the last local scan tick, and does not observe packet delivery. A new scan restarts the schedule. The event-only and always-send controls use the same compact payload selection. Budget is an objective rather than a hard constraint.
* The late-recovery trial separately optimizes changes in waypoint preferences as the deadline approaches. A repeated-scan policy and a route-score eligibility ablation are recorded as validation experiments.

## Reproduce

Run in this directory using the versions in requirements.txt. Existing weights permit evaluation without retraining.

    python -m unittest test_event_radio test_age_policy test_compact_packets
    python age_experiment_policy.py
    python crossed_age_eval.py learned fast
    python crossed_age_eval.py learned bullet

Repeat the last two commands for fixed and blind. Run crossed_age_eval.py transit bullet before running analyze_crossed_age.py. The crossed study reuses each of 120 fresh mission IDs under five radio conditions. Channel effects are paired by mission; 600 rows do not correspond to 600 independent scenarios.

    python compact_packets.py
    python compact_holdout.py
    python compact_full_reference.py
    python sender_schedule.py train
    python sender_schedule.py full
    python sender_schedule.py compact
    python sender_schedule.py once
    python sender_schedule.py learned
    python analyze_sender.py

The sender PyBullet test uses 140 fresh missions crossed with two channel conditions. Bootstrap both conditions together for each sampled mission. The packet novelty pilot and recovery experiment use separate IDs. The trials are alternative project designs, with no claim to literature-wide algorithm priority.

## Results

On the sender holdout (140 distinct scenarios, each repeated under two loss patterns), the eight-record reference and the always-send two-record policy both found 265/280 with mean capped time 13.59 s. Mean transmitted bytes were 3665 and 1678. The trained schedule selected two initial attempts and an eight-tick periodic retry; it found 265/280 in 13.64 s with 1160 transmitted bytes. Event-only transmission used 623 bytes and found 263/280 in 13.99 s. There were zero observed crashes. The paired time difference between the trained schedule and full reference was +0.05 s (95% scenario bootstrap interval -0.15 to +0.34 s). These results apply to this scenario generator and byte-cost setting. A formal noninferiority margin was not prespecified. Files: results/sender-bullet-*.csv and results/communication-search.png.

Learned age decay showed no reproducible advantage over tuned fixed decay. Across five fast-simulator training seeds the burst-versus-scattered interaction averaged -0.105 s (scenario bootstrap interval -0.271 to +0.047). Seed 0 selected identical effective learned and fixed parameters. Shadow-policy tests on 150 matched scenarios found that masking received evidence changed 12.54% of executed action choices in the burst25 setting. This measures decision sensitivity and does not identify a time saving.

Late recovery found 196/200 against 195/200 for the frozen transit policy in the separate PyBullet transfer experiment. Mean capped times were 12.25 and 12.43 s. Both gain intervals included zero. Route eligibility gating underperformed the legacy midpoint score on validation. Learned repeated scans selected the no-repeat control.
