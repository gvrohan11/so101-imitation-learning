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

## MuJoCo PPO: SO-101 ball into cup

The state-based PPO task uses the existing SO-101 MuJoCo model and its actual
finger collision meshes. Run the environment checks before training:

```bash
.venv/bin/python -m experiments.validate_ppo_env
.venv/bin/python -m experiments.train_ppo --smoke-test
```

### Environment contract

- MuJoCo keeps the model's 2 ms timestep (500 Hz). The policy acts at 20 Hz,
  with 25 physics steps per action and a 500-action / 25-second horizon.
- Actions are six normalized values in `[-1, 1]`. The first five move the
  shoulder pan, shoulder lift, elbow, wrist flex, and wrist roll targets by at
  most 2 degrees relative to current measured joint positions. Targets clip to
  the model's joint limits. The sixth changes the gripper target by at most
  0.1 rad per 20 Hz action, relative to its measured joint position, and clips
  to the safe pinch target (`0.26` rad) and open target. Existing position
  servos and force limits remain in use.
- The observation contains 25 physical values: six measured joint positions,
  six measured velocities, gripper XYZ, ball XYZ, cup XYZ, ball linear velocity,
  and elapsed episode fraction. PPO receives normalized simulator state, not
  camera pixels or a scripted one-hot clock.
- Approach shaping follows the demonstrated path: a waypoint 60 mm above the
  grasp site, then the grasp site's offset `(0.020, -0.008, -0.010)` m from
  the ball center with wrist flex `0.5` rad and wrist roll `-2.7` rad. A reach
  event requires the site within 15 mm and both wrist targets within 0.2 rad.
- The arm starts at `[0, 0, 0, 0, 0]` radians with an open gripper. The model
  joint ranges (radians) are shoulder pan `[-1.9199, 1.9199]`, shoulder lift
  `[-1.7453, 1.7453]`, elbow `[-1.69, 1.69]`, wrist flex
  `[-1.6581, 1.6581]`, and wrist roll `[-2.7438, 2.8412]`.
- Stages 1–3 use the fixed 20 mm ball and nominal cup. Stage 2 samples the ball
  within ±10 mm of the verified scene; Stage 3 also samples the cup within
  ±10 mm. Stage 4 adds collision-rejected starting-joint jitter of ±2 degrees.
  Stage 5 currently keeps the ball at 20 mm and samples 0.9, 1.0, or 1.1 cup
  scales. Diagnostic probes at 19, 19.5, 20.5, 21, and 22 mm did not complete
  the actual-gripper transfer reliably, so training does not sample those sizes.
  The narrow XY boxes are centered on the prior successful scripted scene:
  ball `(0.1857, -0.1980)` m and cup `(0.2555, 0.1208)` m.
- Placement must follow a stable bilateral grasp, carry the ball above the cup
  rim while pinched, deliberately open the gripper over the cup, and let the
  ball settle. Success also requires the full ball sphere to fit inside the
  octagonal cup opening with 3 mm clearance, sit between the floor and rim,
  leave all gripper contact, and remain settled for 0.25 seconds. A ball that
  falls or is knocked into the cup without a verified above-rim carry and
  release does not count as a successful placement.

The reward is a sum of signed approach progress (`40` per meter), one-time
grasp-pose reach `+3`, stable-grasp `+2`, signed ball-lift progress (`80` per
meter, capped at the lift threshold), one-time lift `+2`, new-best held carry
height progress (`80` per meter up to 20 mm above the cup rim), new-best held
transport progress (`40` per meter), one-time above-cup `+3`, one-time verified
release `+5`, and terminal success `+50`. It subtracts one-time outside-drop `5`, unsafe collision `0.25`,
joint-limit clipping at `0.1` times the clipped target delta divided by the
2-degree action limit, `0.01` per action step, and unrecoverable failure `10`.
Progress is signed or best-so-far; event bonuses are one-time, so hovering,
oscillating, or repeating grasps cannot farm reward. All weights and thresholds
are in `config/ppo_training.json`.

PPO uses Stable-Baselines3 with randomly initialized `[512, 512]` tanh actor
and critic networks. Training starts from Stage 1 and learns from environment
rewards; the hardcoded expert is not used to initialize, update, evaluate, or
pass the policy. PPO starts with a trainable action standard deviation of
`0.25`, uses `gamma = 0.995` and `GAE lambda = 0.98` to carry credit farther
across the pick-and-place horizon, and keeps entropy regularization enabled.
State-dependent exploration holds coherent action noise for four control
steps, giving the arm time to move in a direction before resampling.
Observation normalization is enabled and reward normalization is off.
Dense approach, grasp-pose, lift, and transport rewards guide PPO before the
terminal placement reward.

Every stage must reach 90% success over 50 evaluation episodes at three
consecutive evaluation windows before the curriculum advances. Stage 1 trains
PPO from scratch on the fixed scene. Stage 2 randomizes the ball, Stage 3 also
randomizes the cup, Stage 4 adds start-joint jitter, and Stage 5 adds cup size
variation. Stage 1 is allowed its full budget even if early evaluations have
no grasps. Later stages retain the zero-grasp early-stop safeguard.

Run a short PPO train/evaluate/checkpoint-reload smoke test, then launch the
full five-stage curriculum:

```bash
.venv/bin/python -m experiments.train_ppo --smoke-test
```

```bash
.venv/bin/python -m experiments.train_ppo
```

The smoke test runs 128 PPO steps and two deterministic evaluation episodes; it
does not launch the curriculum. The full command starts a new, timestamped run
under `outputs/ball_cup_ppo_state_feedback/`, so evaluations cannot be mixed with
older runs. Checkpoints, Monitor episode CSV files, PPO's `progress.csv`, and
deterministic evaluation results are saved there. Each stage is capped at
1,000,000 steps, with evaluations every 25,000 steps after the first 100,000.
Training reward does not advance the curriculum.

The randomization boxes are a conservative ±10 mm neighborhood of one
verified scripted pick-and-place scene, not a complete IK-certified reachable
workspace. Stage 4 checks sampled starts for robot contact at reset. The model
uses the existing 2.7 g ball mass. Broader Stage 5 ball-size randomization is
unresolved because the actual-gripper scripted transfer loses contact at tested
radii other than 20 mm. Real servo velocity limits and camera-derived state
uncertainty are not calibrated yet.

Evaluate the final policy over ten Stage 5 episodes:

```bash
.venv/bin/python -m experiments.evaluate_ppo \
  --model outputs/ball_cup_ppo_anchored/final_model.zip --stage 5 --episodes 10
```

On macOS, record the first episode as a GIF using MuJoCo's CGL renderer:

```bash
MUJOCO_GL=cgl .venv/bin/python -m experiments.evaluate_ppo \
  --model outputs/ball_cup_ppo_anchored/final_model.zip --stage 5 --episodes 5 \
  --video outputs/ball_cup_ppo_anchored/stage5_evaluation.gif
```

# Hardcoded Demo Capability Check

This scripted rollout checks that the MuJoCo scene, gripper, and placement
task are physically solvable. It is a diagnostic only; `experiments.train_ppo`
does not consume its actions or use its success as a curriculum pass.

export MUJOCO_GL=egl
.venv/bin/python -m sim.diagnose_grasp_candidate

## Candidate Video

SO101_RECORD_CANDIDATE_VIDEO=1 .venv/bin/python -m sim.diagnose_grasp_candidate

# Running PPO Training

export MUJOCO_GL=egl
.venv/bin/python -m experiments.train_ppo
