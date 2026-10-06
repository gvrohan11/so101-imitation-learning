# LeWM visual policy for SO-101 ball-in-cup

This is an isolated vision-policy experiment. It leaves the existing state-only PPO implementation unchanged.

## What the policy observes

- A 224×224 RGB image from the MuJoCo front camera.
- Six measured joint positions, six measured joint velocities, and elapsed episode fraction.
- The same six normalized motor actions as the existing environment: 20 Hz control and a maximum 2° arm-joint change per action.

Ball and cup world coordinates are deliberately excluded from the policy observation. MuJoCo still uses the configured physical task, reward, and success checks.

## How LeWM is used

The default `lewm` backbone loads the existing `lewm_seq_projectors.pt` checkpoint and freezes its image encoder and trained projection head. PPO trains the state branch, fusion layer, actor, and critic. The action-conditioned predictor is not used to choose imagined actions: its current transition results did not establish an advantage over persistence. This experiment tests whether the checkpoint's visual representation helps this task; it does not retrain LeWM.

The inference-time definitions matching the LeWM encoder and projection head are included in this folder. RunPod does not need a clone of the LeWM source repository; it only needs the `.pt` checkpoint. For a controlled feature comparison, the same trainer accepts `--visual-backbone resnet18`. Both variants get the same camera images, robot-state fields, demonstrations, PPO settings, and evaluation seeds. ResNet defaults to ImageNet weights; use `--random-resnet` only for an explicitly untrained visual baseline.

## Run on RunPod

From the SO-101 repository root, point at the checkpoint file. On a headless Linux node, MuJoCo normally needs EGL:

```bash
export MUJOCO_GL=egl
CHECKPOINT=/workspace/checkpoints/lewm_seq_projectors.pt
```

Upload or copy the 207 MB `lewm_seq_projectors.pt` file to that path. The model code itself is bundled here, so there is no required `LEWM_REPO` path.

First verify that an actual MuJoCo camera frame passes through the saved LeWM weights and yields finite 192-dimensional features. This does not train anything:

```bash
.venv/bin/python -m experiments.lewm_policy.check_encoder \
  --checkpoint "$CHECKPOINT" --device cuda
```

If running from a headless Mac shell that cannot create a CoreGraphics/OpenGL
context, the LeWM weights can still be checked against the saved simulator
camera frame in this repository:

```bash
.venv/bin/python -m experiments.lewm_policy.check_encoder \
  --checkpoint "$CHECKPOINT" \
  --image results/ball_cup_camera.png --device cpu
```

Recording a fresh scripted demo and running PPO require a working MuJoCo renderer;
on a headless Linux machine set `MUJOCO_GL=egl` as above.

Record eight successful scripted episodes with aligned RGB frames, robot state, and actions. The teacher lightly perturbs executed arm commands and replans from the resulting state, while saving its clean corrective commands as labels. This covers small trajectory deviations that a single clean demonstration cannot teach. Then initialize PPO's actor from these examples and train the LeWM policy:

```bash
.venv/bin/python -m experiments.lewm_policy.record_demo \
  --output outputs/lewm_policy/expert_demo.npz \
  --episodes 8 --execution-action-noise-std 0.1

.venv/bin/python -m experiments.lewm_policy.train \
  --visual-backbone lewm --checkpoint "$CHECKPOINT" \
  --demo outputs/lewm_policy/expert_demo.npz \
  --stage 1 --timesteps 250000 --device cuda
```

Stage 1 reproduces the fixed scene used by the successful scripted demonstration. Once that works, use `--stage 2` to randomize the ball position while keeping the cup fixed. Higher stages randomize more of the setup. The evaluation callback saves per-episode phase rates and best/latest model checkpoints under `outputs/lewm_policy/lewm/stage_XX/`.

The training command does not require TensorBoard. Before PPO starts, it behavior-clones the state/fusion/actor layers from the recorded actions and replays the policy on the fixed scene. PPO is deliberately stopped unless that warm-start replay reaches the configured minimum success rate (90% by default). Add `--refresh-demo` to overwrite an existing demo with the multi-episode corrected set. If the gate still stops, the run prints phase rates and per-episode failure reasons, and saves them to `initialization.json`.

Run the ResNet comparison with the same demonstration and stage:

```bash
.venv/bin/python -m experiments.lewm_policy.train \
  --visual-backbone resnet18 --demo outputs/lewm_policy/expert_demo.npz \
  --stage 1 --timesteps 250000 --device cuda
```

The checkpoint is read from the path you pass; this folder does not copy the 207 MB file into Git. Supply its path again when loading/running the LeWM policy on another machine.
