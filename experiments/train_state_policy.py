import argparse
import json
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.gym_env import StateOnlyBallCupEnv


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def distribution(self, observations):
        mean = self.actor(observations)
        standard_deviation = self.log_std.exp().expand_as(mean)
        return Normal(mean, standard_deviation)

    @staticmethod
    def _squashed_log_probability(distribution, raw_actions):
        actions = torch.tanh(raw_actions)
        correction = torch.log(1.0 - actions.square() + 1e-6)
        return distribution.log_prob(raw_actions).sum(dim=-1) - correction.sum(
            dim=-1
        )

    def act(self, observations, deterministic=False):
        distribution = self.distribution(observations)
        raw_actions = (
            distribution.mean if deterministic else distribution.sample()
        )
        log_probability = self._squashed_log_probability(
            distribution, raw_actions
        )
        values = self.critic(observations).squeeze(-1)
        return torch.tanh(raw_actions), raw_actions, log_probability, values

    def evaluate_actions(self, observations, raw_actions):
        distribution = self.distribution(observations)
        log_probabilities = self._squashed_log_probability(
            distribution, raw_actions
        )
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(observations).squeeze(-1)
        return log_probabilities, entropy, values


def evaluate_policy(policy, episodes, seed, device, horizon):
    env = StateOnlyBallCupEnv(horizon=horizon)
    episode_returns = []
    successes = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                state = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = policy.act(state, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                episode_return += reward
                if terminated or truncated:
                    episode_returns.append(episode_return)
                    successes.append(bool(info["is_success"]))
                    break
    finally:
        env.close()
    return float(np.mean(episode_returns)), float(np.mean(successes))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = StateOnlyBallCupEnv(horizon=args.horizon, seed=args.seed)
    policy = ActorCritic(
        env.observation_space.shape[0], env.action_space.shape[0]
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    observation, _ = env.reset(seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_success_rate = -1.0
    total_steps = 0

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            while total_steps < args.total_steps:
                previous_total_steps = total_steps
                rollout_size = min(args.rollout_steps, args.total_steps - total_steps)
                observations = []
                raw_actions = []
                log_probabilities = []
                values = []
                next_values = []
                rewards = []
                terminated_flags = []
                episode_boundaries = []
                episode_returns = []
                current_return = 0.0

                for _ in range(rollout_size):
                    state_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action, raw_action, log_probability, value = policy.act(
                            state_tensor
                        )
                    next_observation, reward, terminated, truncated, _ = env.step(
                        action.squeeze(0).cpu().numpy()
                    )
                    next_state_tensor = torch.as_tensor(
                        next_observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        next_value = policy.critic(next_state_tensor).squeeze()

                    observations.append(observation)
                    raw_actions.append(raw_action.squeeze(0).cpu().numpy())
                    log_probabilities.append(log_probability.item())
                    values.append(value.item())
                    next_values.append(next_value.item())
                    rewards.append(float(reward))
                    terminated_flags.append(float(terminated))
                    episode_boundaries.append(float(terminated or truncated))
                    current_return += reward
                    observation = next_observation
                    total_steps += 1

                    if terminated or truncated:
                        episode_returns.append(current_return)
                        current_return = 0.0
                        observation, _ = env.reset()

                obs_tensor = torch.as_tensor(
                    np.asarray(observations), dtype=torch.float32, device=device
                )
                raw_action_tensor = torch.as_tensor(
                    np.asarray(raw_actions), dtype=torch.float32, device=device
                )
                old_log_probability_tensor = torch.as_tensor(
                    log_probabilities, dtype=torch.float32, device=device
                )
                value_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
                next_value_tensor = torch.as_tensor(
                    next_values, dtype=torch.float32, device=device
                )
                reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                terminated_tensor = torch.as_tensor(
                    terminated_flags, dtype=torch.float32, device=device
                )
                boundary_tensor = torch.as_tensor(
                    episode_boundaries, dtype=torch.float32, device=device
                )

                advantages = torch.zeros_like(reward_tensor)
                gae = torch.tensor(0.0, device=device)
                for index in reversed(range(rollout_size)):
                    delta = (
                        reward_tensor[index]
                        + args.gamma * next_value_tensor[index]
                        * (1.0 - terminated_tensor[index])
                        - value_tensor[index]
                    )
                    gae = delta + args.gamma * args.gae_lambda * (
                        1.0 - boundary_tensor[index]
                    ) * gae
                    advantages[index] = gae
                returns = advantages + value_tensor
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

                indices = np.arange(rollout_size)
                for _ in range(args.update_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, rollout_size, args.minibatch_size):
                        batch_indices = indices[start:start + args.minibatch_size]
                        batch = torch.as_tensor(batch_indices, device=device)
                        new_log_probabilities, entropy, new_values = (
                            policy.evaluate_actions(
                                obs_tensor[batch], raw_action_tensor[batch]
                            )
                        )
                        ratio = (
                            new_log_probabilities
                            - old_log_probability_tensor[batch]
                        ).exp()
                        unclipped = ratio * advantages[batch]
                        clipped = torch.clamp(
                            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                        ) * advantages[batch]
                        policy_loss = -torch.minimum(unclipped, clipped).mean()
                        value_loss = 0.5 * (
                            new_values - returns[batch]
                        ).square().mean()
                        loss = (
                            policy_loss
                            + args.value_coefficient * value_loss
                            - args.entropy_coefficient * entropy.mean()
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                        optimizer.step()

                crossed_eval_interval = (
                    total_steps // args.eval_every
                    > previous_total_steps // args.eval_every
                )
                if crossed_eval_interval or total_steps >= args.total_steps:
                    mean_return, success_rate = evaluate_policy(
                        policy,
                        args.eval_episodes,
                        args.seed + total_steps,
                        device,
                        args.horizon,
                    )
                    record = {
                        "steps": total_steps,
                        "mean_train_return": (
                            float(np.mean(episode_returns))
                            if episode_returns else None
                        ),
                        "eval_mean_return": mean_return,
                        "eval_success_rate": success_rate,
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    print(json.dumps(record), flush=True)
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        torch.save(
                            {
                                "policy": policy.state_dict(),
                                "observation_size": env.observation_space.shape[0],
                                "action_size": env.action_space.shape[0],
                                "steps": total_steps,
                                "eval_success_rate": success_rate,
                            },
                            output_dir / "best_policy.pt",
                        )

        torch.save(
            {
                "policy": policy.state_dict(),
                "observation_size": env.observation_space.shape[0],
                "action_size": env.action_space.shape[0],
                "steps": total_steps,
            },
            output_dir / "final_policy.pt",
        )
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PPO policy from simulator state."
    )
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2_048)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/state_policy")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
