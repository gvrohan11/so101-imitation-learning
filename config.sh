#!/usr/bin/env bash
# Edit these once, then every script below just works.
export HF_USER="your-hf-username"
export TASK="Pick up the red cube and drop it in the cup"
export TASK_SLUG="cube_in_cup"

# Serial ports — macOS: /dev/tty.usbmodemXXXX (find with: ls /dev/tty.usbmodem*)
# Or run: lerobot-find-port
export FOLLOWER_PORT="/dev/ttyACM0"
export LEADER_PORT="/dev/ttyACM1"
export FOLLOWER_ID="so101_follower_arm"
export LEADER_ID="so101_leader_arm"

# Cameras — wrist is the important one; add a second if you have a spare USB cam
export CAMERAS='{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }'

# Recording defaults
export NUM_EPISODES=50
export EPISODE_TIME_S=20
export RESET_TIME_S=10

# Training
export TRAIN_STEPS=100000
export BATCH_SIZE=8
export DEVICE="cuda"   # "cuda" on a GPU box, "mps" on your Mac, "cpu" = don't
