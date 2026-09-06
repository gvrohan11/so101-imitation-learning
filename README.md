# so101-imitation-learning
so101-imitation-learning

Here's the first project, start to finish.

Level 1: make it work. Teleoperate 50 rounds of cube in cup task. Train an ACT policy on this. Then, the arm should do this task autonomously

Level 2: reuse this info. Train on 10,20,40,80 demos. Plot success vs # of demos.
ACT vs diff policy - same data, 2 policies, head-to-head success rate with confidence intervals

Level 3 - fine-tune smolVLA (small VLA model) on collected data. This means that we would have fine tuned a small LM, then a small VLA on the data

For the arm: find port -> calibrate -> teleop

The data we collect comes from each teleop round: Camera frames, Robot state, Action:
One demo (ex: 15 seconds of dropping the cube in the cup) = hundreds of synchronized (image, state, action) snapshots


# Status
Validated the full training pipeline in simulation (ACT on the pusht dataset, trained on Apple MPS) before touching hardware — confirming the record→train→eval→plot loop works end to end. Real SO-101 data slots into the same pipeline unchanged.