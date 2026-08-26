from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from pokemon_battler.data.training_data import InteractionInferenceCollator


ROLLOUT_SCHEMA = "qwen-ppo-rollout-v1"


class PPORolloutCollator(InteractionInferenceCollator):
    """Collate saved live observations plus their on-policy PPO statistics."""

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if any(row.get("rollout_schema") != ROLLOUT_SCHEMA for row in rows):
            raise ValueError(f"PPO rows must use {ROLLOUT_SCHEMA}")
        batch = super().__call__(rows)
        batch.update(
            {
                "action_ids": torch.tensor(
                    [int(row["action_id"]) for row in rows], dtype=torch.long
                ),
                "old_log_probs": torch.tensor(
                    [float(row["old_log_probability"]) for row in rows],
                    dtype=torch.float32,
                ),
                "old_values": torch.tensor(
                    [float(row["old_value"]) for row in rows], dtype=torch.float32
                ),
                "advantages": torch.tensor(
                    [float(row["advantage"]) for row in rows], dtype=torch.float32
                ),
                "returns": torch.tensor(
                    [float(row["return"]) for row in rows], dtype=torch.float32
                ),
            }
        )
        return batch


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    expectile: float,
) -> torch.Tensor:
    """Asymmetric squared loss used for the IQL state-value fit."""
    if not 0 < expectile < 1:
        raise ValueError("expectile must be between zero and one")
    difference = target - prediction
    weights = torch.where(difference >= 0, expectile, 1.0 - expectile)
    return (weights * difference.square()).mean()


def offline_outcome_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    expectile: float = 0.7,
    advantage_temperature: float = 0.1,
    max_advantage_weight: float = 20.0,
    behavior_clone_weight: float = 0.1,
    q_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Outcome-conditioned, IQL-style loss for logged replay decisions.

    The replay archive contains a final battle result but not counterfactual
    next states for actions the player did not take.  The selected action's
    Monte-Carlo return is therefore the exact available Q target.  The state
    value uses IQL expectile regression and the policy uses advantage-weighted
    behavior cloning.
    """
    if advantage_temperature <= 0:
        raise ValueError("advantage_temperature must be positive")
    if max_advantage_weight < 1:
        raise ValueError("max_advantage_weight must be at least one")

    action_log_probs = outputs["action_log_probs"].float()
    action_ids = batch["action_ids"].to(action_log_probs.device)
    selected_log_probs = action_log_probs.gather(1, action_ids[:, None]).squeeze(1)
    if not bool(torch.isfinite(selected_log_probs).all()):
        raise ValueError("An offline target is absent from its legal-action mask")

    outcome_targets = batch["value_targets"].to(action_log_probs.device).float()
    valid = outcome_targets >= 0
    if not bool(valid.any()):
        raise ValueError("Offline outcome training requires WIN/LOSS labels")

    selected_q_logits = outputs["action_value_logits"].float().gather(
        1, action_ids[:, None]
    ).squeeze(1)
    selected_q = torch.sigmoid(selected_q_logits)
    state_value = torch.sigmoid(outputs["value_logits"].float())
    row_weights = batch.get("value_weights")
    if row_weights is None:
        row_weights = torch.ones_like(outcome_targets)
    else:
        row_weights = row_weights.to(action_log_probs.device).float()
    normalized_row_weights = row_weights[valid] / row_weights[valid].mean().clamp_min(1e-8)
    q_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_q_logits[valid], outcome_targets[valid], reduction="none"
    )
    q_loss = (normalized_row_weights * q_losses).mean()
    value_difference = selected_q.detach()[valid] - state_value[valid]
    expectile_weights = torch.where(
        value_difference >= 0, expectile, 1.0 - expectile
    )
    v_loss = (
        normalized_row_weights * expectile_weights * value_difference.square()
    ).mean()

    advantages = selected_q.detach() - state_value.detach()
    advantage_weights = torch.exp(advantages / advantage_temperature).clamp(
        max=max_advantage_weight
    )
    policy_weights = normalized_row_weights * advantage_weights[valid]
    weighted_policy_loss = -(
        policy_weights * selected_log_probs[valid]
    ).sum() / policy_weights.sum().clamp_min(1e-8)
    behavior_clone_loss = -(
        normalized_row_weights * selected_log_probs[valid]
    ).mean()
    total = (
        weighted_policy_loss
        + behavior_clone_weight * behavior_clone_loss
        + q_weight * q_loss
        + value_weight * v_loss
    )
    return total, {
        "offline_policy_loss": weighted_policy_loss.detach(),
        "behavior_clone_loss": behavior_clone_loss.detach(),
        "q_loss": q_loss.detach(),
        "expectile_value_loss": v_loss.detach(),
        "mean_advantage": advantages[valid].mean().detach(),
        "mean_advantage_weight": advantage_weights[valid].mean().detach(),
        "q_accuracy": (
            (selected_q[valid] >= 0.5) == (outcome_targets[valid] >= 0.5)
        ).float().mean().detach(),
    }


def generalized_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute GAE and value returns for one actor trajectory."""
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have the same length")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    advantages = [0.0] * len(rewards)
    running_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        running_advantage = (
            delta
            + gamma * gae_lambda * continuation * running_advantage
        )
        advantages[index] = running_advantage
        next_value = values[index]
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def ppo_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    clip_ratio: float = 0.2,
    value_clip: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Legal-masked PPO objective over Qwen's direct action distribution."""
    if clip_ratio <= 0 or value_clip < 0:
        raise ValueError("PPO clipping values must be non-negative")
    log_probs = outputs["action_log_probs"].float()
    action_ids = batch["action_ids"].to(log_probs.device)
    new_log_probs = log_probs.gather(1, action_ids[:, None]).squeeze(1)
    old_log_probs = batch["old_log_probs"].to(log_probs.device).float()
    advantages = batch["advantages"].to(log_probs.device).float()

    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    # The retained checkpoint value head is a win probability.  Centering it
    # maps the prediction to the same [-1, 1] scale as terminal battle rewards.
    new_values = 2.0 * torch.sigmoid(outputs["value_logits"].float()) - 1.0
    old_values = batch["old_values"].to(log_probs.device).float()
    returns = batch["returns"].to(log_probs.device).float()
    unclipped_value_error = (new_values - returns).square()
    clipped_values = old_values + (new_values - old_values).clamp(
        -value_clip, value_clip
    )
    clipped_value_error = (clipped_values - returns).square()
    value_loss = 0.5 * torch.maximum(
        unclipped_value_error, clipped_value_error
    ).mean()

    legal = torch.isfinite(log_probs)
    probabilities = torch.where(legal, log_probs.exp(), torch.zeros_like(log_probs))
    safe_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    entropy = -(probabilities * safe_log_probs).sum(dim=1).mean()
    total = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    with torch.no_grad():
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    return total, {
        "ppo_policy_loss": policy_loss.detach(),
        "ppo_value_loss": value_loss.detach(),
        "policy_entropy": entropy.detach(),
        "approximate_kl": approximate_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "mean_value": new_values.mean().detach(),
        "mean_return": returns.mean().detach(),
    }


class WinTrajectoryBuffer:
    """Thread-safe conversion of live decisions into completed PPO rows."""

    def __init__(self, *, gamma: float = 1.0, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._completed: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_decision(
        self,
        battle_id: str,
        observation: dict[str, Any],
        *,
        action_id: int,
        old_log_probability: float,
        value_probability: float,
    ) -> None:
        if not math.isfinite(old_log_probability):
            raise ValueError("old action log probability must be finite")
        if not 0 <= value_probability <= 1:
            raise ValueError("value probability must be in [0, 1]")
        with self._lock:
            self._pending[battle_id].append(
                {
                    "observation": observation,
                    "action_id": int(action_id),
                    "old_log_probability": float(old_log_probability),
                    "old_value": 2.0 * float(value_probability) - 1.0,
                }
            )

    def finish_battle(self, battle_id: str, *, won: bool, lost: bool) -> None:
        if won and lost:
            raise ValueError("A battle cannot be both won and lost")
        reward = 1.0 if won else -1.0 if lost else 0.0
        with self._lock:
            decisions = self._pending.pop(battle_id, [])
            if not decisions:
                return
            rewards = [0.0] * len(decisions)
            rewards[-1] = reward
            dones = [False] * len(decisions)
            dones[-1] = True
            advantages, returns = generalized_advantages(
                rewards,
                [float(item["old_value"]) for item in decisions],
                dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            for index, (decision, advantage, value_return) in enumerate(
                zip(decisions, advantages, returns)
            ):
                self._completed.append(
                    decision["observation"]
                    | {
                        "rollout_schema": ROLLOUT_SCHEMA,
                        "rollout_battle_id": battle_id,
                        "rollout_index": index,
                        "action_id": decision["action_id"],
                        "old_log_probability": decision["old_log_probability"],
                        "old_value": decision["old_value"],
                        "reward": rewards[index],
                        "done": dones[index],
                        "advantage": advantage,
                        "return": value_return,
                        "outcome": "WIN" if won else "LOSS" if lost else "TIE",
                    }
                )

    @property
    def pending_battles(self) -> int:
        with self._lock:
            return len(self._pending)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._completed)

    def write_jsonl(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        if rows:
            raw_advantages = torch.tensor(
                [float(row["advantage"]) for row in rows], dtype=torch.float64
            )
            advantage_mean = float(raw_advantages.mean().item())
            advantage_std = float(raw_advantages.std(unbiased=False).item())
            if advantage_std > 1e-8:
                for row in rows:
                    row["raw_advantage"] = row["advantage"]
                    row["advantage"] = (
                        float(row["advantage"]) - advantage_mean
                    ) / advantage_std
        else:
            advantage_mean = 0.0
            advantage_std = 0.0
        battle_ids = {str(row["rollout_battle_id"]) for row in rows}
        with destination.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
                stream.write("\n")
        return {
            "schema": ROLLOUT_SCHEMA,
            "path": str(destination),
            "battles": len(battle_ids),
            "decisions": len(rows),
            "pending_battles": self.pending_battles,
            "raw_advantage_mean": advantage_mean,
            "raw_advantage_std": advantage_std,
        }
