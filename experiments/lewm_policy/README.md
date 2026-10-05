# LeWM visual policy for SO-101 ball-in-cup

This is an isolated vision-policy experiment. It leaves the existing state-only PPO implementation unchanged.

## What the policy observes

- A 224×224 RGB image from the MuJoCo front camera.
- Six measured joint positions, six measured joint velocities, and elapsed episode fraction.
- The same six normalized motor actions as the existing environment: 20 Hz control and a maximum 2° arm-joint change per action.

Ball and cup world coordinates are deliberately excluded from the policy observation. MuJoCo still uses the configured physical task, reward, and success checks.

## How LeWM is used

The default `lewm` backbone loads the existing `lewm_seq_projectors.pt` checkpoint and freezes its image encoder and trained projection head. PPO trains the state branch, fusion layer, actor, and critic. The action-conditioned predictor is not used to choose imagined actions: its current transition results did not establish an advantage over persistence. This experiment tests whether the checkpoint's visual representation helps this task; it does not retrain LeWM.

For a controlled feature comparison, the same trainer accepts `--visual-backbone resnet18`. Both variants get the same camera images, robot-state fields, demonstrations, PPO settings, and evaluation seeds. ResNet defaults to ImageNet weights; use `--random-resnet` only for an explicitly untrained visual baseline.

## Run on RunPod

From the SO-101 repository root, point at the LeWM checkout and checkpoint. On a headless Linux node, MuJoCo normally needs EGL:

```bash
export MUJOCO_GL=egl
LEWM_REPO=/workspace/Le-World-Model-Implementation
CHECKPOINT="$LEWM_REPO/lewm_seq_projectors.pt"
```

First verify that an actual MuJoCo camera frame passes through the saved LeWM weights and yields finite 192-dimensional features. This does not train anything:

```bash
.venv/bin/python -m experiments.lewm_policy.check_encoder \
  --lewm-repo "$LEWM_REPO" --checkpoint "$CHECKPOINT" --device cuda
```

If running from a headless Mac shell that cannot create a CoreGraphics/OpenGL
context, the LeWM weights can still be checked against the saved simulator
camera frame in this repository:

```bash
.venv/bin/python -m experiments.lewm_policy.check_encoder \
  --lewm-repo "$LEWM_REPO" --checkpoint "$CHECKPOINT" \
  --image results/ball_cup_camera.png --device cpu
```

Recording a fresh scripted demo and running PPO require a working MuJoCo renderer;
on a headless Linux machine set `MUJOCO_GL=egl` as above.

Record a successful scripted episode with aligned RGB frames, robot state, and actions, then initialize PPO's actor from that example and train the LeWM policy:

```bash
.venv/bin/python -m experiments.lewm_policy.record_demo \
  --output outputs/lewm_policy/expert_demo.npz

.venv/bin/python -m experiments.lewm_policy.train \
  --visual-backbone lewm --lewm-repo "$LEWM_REPO" \
  --checkpoint "$CHECKPOINT" --demo outputs/lewm_policy/expert_demo.npz \
  --stage 1 --timesteps 250000 --device cuda
```

Stage 1 reproduces the fixed scene used by the successful scripted demonstration. Once that works, use `--stage 2` to randomize the ball position while keeping the cup fixed. Higher stages randomize more of the setup. The evaluation callback saves per-episode phase rates and best/latest model checkpoints under `outputs/lewm_policy/lewm/stage_XX/`.

Run the ResNet comparison with the same demonstration and stage:

```bash
.venv/bin/python -m experiments.lewm_policy.train \
  --visual-backbone resnet18 --demo outputs/lewm_policy/expert_demo.npz \
  --stage 1 --timesteps 250000 --device cuda
```

The checkpoint remains in the LeWM repository; this folder does not copy the 207 MB file into Git. Supply its path again when loading/running the LeWM policy on another machine.
