# so101-imitation-learning
so101-imitation-learning

Here's the first project, start to finish.

The setup (once): Assemble/plug in the SO-101, calibrate both arms, confirm the follower mirrors the leader when you puppet it. Your M2 drives the arm; a cloud GPU does the training.

The task: Pick up a red cube, drop it in a cup. Deliberately simple, with a crisp success/fail moment so the numbers mean something.

The three levels — all one project, one arm, one dataset:

Level 1 - make it work. Teleoperate ~50 demonstrations of cube-in-cup with a wrist camera, which log to a LeRobotDataset. Train an ACT policy on them. The arm now does the task autonomously — no hardcoded motion, it learned from your demos. This alone is a complete project.

Level 2 — make it a study (this is the target). Reuse that same dataset, no new data collection:
Data-efficiency curve — train on 10 / 20 / 40 / 80 demos, plot success rate vs. number of demos. How much data does the task actually need?
ACT vs. diffusion policy — same data, two policy types, head-to-head success rate with confidence intervals + a note on failure modes.
This is what turns "I ran the tutorial" into "I ran an experiment." Get here and it's genuinely strong.

Level 3 — the stretch (optional). Fine-tune SmolVLA (a small vision-language-action model) on your own collected data. Ties to your SmolLM2 work: "I fine-tuned a small LM, then a small VLA on data from a robot on my desk."

The deliverable: A clean public repo (the one I already built — record/train/eval scripts, the eval harness with Wilson CIs, the plotting), a short README framing it as an experiment, one video of the arm succeeding on its own, and the dataset + policy pushed to the HF Hub.

Why it's scoped for ~a week: You collect data once; everything after is compute, not labor. The repo's already wired for exactly this flow.

Whenever the arm arrives, we start at find-port → calibrate → teleop, then record that first dataset.