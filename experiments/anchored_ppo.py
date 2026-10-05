"""PPO with a supervised projection that preserves expert behavior."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


class DemonstrationAnchoredPPO(PPO):
    """Apply PPO updates, then pull the actor back toward the expert trajectory."""

    def set_demonstration_anchor(
        self,
        observations,
        actions,
        *,
        ridge: float = 1e-4,
    ) -> None:
        self._anchor_observations = np.asarray(observations, dtype=np.float32).copy()
        self._anchor_actions = np.asarray(actions, dtype=np.float32).copy()
        if self._anchor_observations.ndim != 2 or self._anchor_actions.ndim != 2:
            raise ValueError("Demonstration observations and actions must be matrices")
        if len(self._anchor_observations) != len(self._anchor_actions):
            raise ValueError("Demonstration observation/action lengths differ")
        if not len(self._anchor_actions):
            raise ValueError("Cannot anchor PPO to an empty demonstration")
        self._anchor_ridge = float(ridge)
        if self._anchor_ridge < 0.0:
            raise ValueError("Demonstration projection ridge must be nonnegative")

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "_anchor_observations",
            "_anchor_actions",
        ]

    def train(self) -> None:
        super().train()
        if not hasattr(self, "_anchor_observations"):
            return

        observations = self._anchor_observations
        vec_normalize = self.get_vec_normalize_env()
        if vec_normalize is not None:
            observations = vec_normalize.normalize_obs(observations)
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self._anchor_actions, dtype=torch.float32, device=self.device
        )

        # PPO can move the actor far in a single minibatch. Project just its
        # linear readout to match the expert actions on the demonstration states
        # while making the smallest possible parameter change. This is one
        # small ridge solve, not dozens of optimizer passes, and it leaves the
        # hidden state-feedback features available for randomized scenes.
        with torch.no_grad():
            self.policy.set_training_mode(False)
            features = self.policy.extract_features(
                observations, self.policy.pi_features_extractor
            )
            latent = self.policy.mlp_extractor.forward_actor(features)
            design = torch.cat(
                (latent, torch.ones((len(latent), 1), device=self.device)), dim=1
            )
            design64 = design.to(torch.float64)
            target64 = actions.to(torch.float64)
            action_head = self.policy.action_net
            old_parameters = torch.cat(
                (action_head.weight.T, action_head.bias.unsqueeze(0)), dim=0
            ).to(torch.float64)
            residual = target64 - design64 @ old_parameters
            gram = design64 @ design64.T
            if self._anchor_ridge:
                gram = gram + self._anchor_ridge * torch.eye(
                    len(gram), dtype=gram.dtype, device=gram.device
                )
            adjustment = design64.T @ torch.linalg.solve(gram, residual)
            projected = old_parameters + adjustment
            action_head.weight.copy_(projected[:-1].T.to(action_head.weight.dtype))
            action_head.bias.copy_(projected[-1].to(action_head.bias.dtype))
            final_mean = self.policy.get_distribution(observations).distribution.mean
            final_mse = F.mse_loss(final_mean, actions).item()
        self.logger.record(
            "train/demonstration_action_mse",
            float(final_mse),
        )
