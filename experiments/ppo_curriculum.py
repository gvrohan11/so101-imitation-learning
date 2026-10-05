"""Success-evaluation based PPO curriculum callbacks."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from experiments.ppo_common import run_deterministic_evaluation, save_bundle, write_json_line


class CurriculumStageCallback(BaseCallback):
    """Stop a stage only after repeated deterministic success evaluations."""

    def __init__(
        self,
        *,
        stage: int,
        output_dir: str | Path,
        seed: int,
        minimum_steps: int,
        evaluation_interval: int,
        evaluation_episodes: int,
        required_success_rate: float,
        required_consecutive: int,
        maximum_steps: int,
        checkpoint_interval: int,
        global_best: dict,
        maximum_consecutive_zero_grasp_evaluations: int = 3,
        config_path=None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.stage = int(stage)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.minimum_steps = int(minimum_steps)
        self.evaluation_interval = int(evaluation_interval)
        self.evaluation_episodes = int(evaluation_episodes)
        self.required_success_rate = float(required_success_rate)
        self.required_consecutive = int(required_consecutive)
        self.maximum_steps = int(maximum_steps)
        self.checkpoint_interval = int(checkpoint_interval)
        self.maximum_consecutive_zero_grasp_evaluations = int(
            maximum_consecutive_zero_grasp_evaluations
        )
        self.global_best = global_best
        self.config_path = config_path
        self.stage_start_steps = 0
        self.last_checkpoint_steps = 0
        self.evaluation_index = 0
        self.consecutive_passes = 0
        self.consecutive_zero_grasp_evaluations = 0
        self.best_stage_success_rate = -1.0
        self.outcome = "running"

    def _on_training_start(self) -> None:
        self.stage_start_steps = int(self.model.num_timesteps)

    def _save(self, label: str) -> None:
        stage_dir = self.output_dir / f"stage_{self.stage:02d}"
        save_bundle(self.model, stage_dir, label)
        save_bundle(self.model, self.output_dir, "latest_model")

    def _on_step(self) -> bool:
        stage_steps = int(self.num_timesteps - self.stage_start_steps)

        if (
            stage_steps > 0
            and stage_steps % self.checkpoint_interval == 0
            and stage_steps != self.last_checkpoint_steps
        ):
            self.last_checkpoint_steps = stage_steps
            self._save(f"checkpoint_{stage_steps:08d}")

        should_evaluate = (
            stage_steps >= self.minimum_steps
            and self.evaluation_interval > 0
            and stage_steps % self.evaluation_interval == 0
        )
        if should_evaluate:
            self.evaluation_index += 1
            eval_seed = (
                self.seed
                + self.stage * 10_000_000
                + self.evaluation_index * self.evaluation_episodes
            )
            result = run_deterministic_evaluation(
                self.model,
                self.model.get_vec_normalize_env(),
                stage=self.stage,
                episodes=self.evaluation_episodes,
                seed=eval_seed,
                config_path=self.config_path,
            )
            result.update({
                "global_env_steps": int(self.num_timesteps),
                "stage_steps": stage_steps,
                "evaluation_index": self.evaluation_index,
                "consecutive_passes_before_this_eval": self.consecutive_passes,
            })
            write_json_line(self.output_dir / "evaluations.jsonl", result)

            success_rate = result["success_rate"]
            grasp_rate = result["phase_rates"].get("grasp_success", 0.0)
            if grasp_rate <= 0.0:
                self.consecutive_zero_grasp_evaluations += 1
            else:
                self.consecutive_zero_grasp_evaluations = 0
            if success_rate >= self.required_success_rate:
                self.consecutive_passes += 1
            else:
                self.consecutive_passes = 0

            if success_rate > self.best_stage_success_rate:
                self.best_stage_success_rate = success_rate
                save_bundle(
                    self.model,
                    self.output_dir / f"stage_{self.stage:02d}",
                    "best_model",
                )
            if (
                success_rate > self.global_best["success_rate"]
                or (
                    success_rate == self.global_best["success_rate"]
                    and self.stage > self.global_best["stage"]
                )
            ):
                self.global_best.update({
                    "success_rate": success_rate,
                    "stage": self.stage,
                    "global_env_steps": int(self.num_timesteps),
                })
                save_bundle(self.model, self.output_dir, "best_model")

            if self.verbose:
                print(
                    f"Stage {self.stage} evaluation {self.evaluation_index}: "
                    f"{result['successes']}/{self.evaluation_episodes} "
                    f"successes ({success_rate:.1%}); "
                    f"passing streak {self.consecutive_passes}/"
                    f"{self.required_consecutive}"
                )

            if self.consecutive_passes >= self.required_consecutive:
                self.outcome = "passed"
                self._save("latest_model")
                return False

            if (
                self.maximum_consecutive_zero_grasp_evaluations > 0
                and self.consecutive_zero_grasp_evaluations
                >= self.maximum_consecutive_zero_grasp_evaluations
            ):
                self.outcome = "failed_no_grasp_progress"
                self._save("latest_model")
                if self.verbose:
                    print(
                        f"Stopping Stage {self.stage} after "
                        f"{self.consecutive_zero_grasp_evaluations} consecutive "
                        "evaluations with no stable grasp."
                    )
                return False

        if stage_steps >= self.maximum_steps:
            self.outcome = "failed_maximum_budget"
            self._save("latest_model")
            return False
        return True
