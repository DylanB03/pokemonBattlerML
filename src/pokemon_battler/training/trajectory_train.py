from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pokemon_battler.data.frozen_cache import checkpoint_signature
from pokemon_battler.training.train import set_seed
from pokemon_battler.data.trajectory_cache import EncodedTrajectoryCache
from pokemon_battler.models.trajectory_modeling import (
    TrajectoryPolicyHead,
    save_trajectory_head,
)


class SequenceWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose every trajectory turn once for loss, with recurrent burn-in and lookahead."""

    def __init__(
        self,
        cache: EncodedTrajectoryCache,
        *,
        sequence_length: int = 64,
        burn_in: int = 16,
    ) -> None:
        if sequence_length <= 0 or burn_in < 0:
            raise ValueError("sequence_length must be positive and burn_in non-negative")
        self.cache = cache
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.windows: list[tuple[int, int, int, int]] = []
        for span_index, span in enumerate(cache.spans):
            start = int(span["start"])
            end = int(span["end"])
            if end <= start:
                raise ValueError("Trajectory cache contains an empty span")
            train_start = start
            while train_start < end:
                train_end = min(train_start + sequence_length, end)
                window_start = max(start, train_start - burn_in)
                fetch_end = min(train_end + 1, end)
                self.windows.append((span_index, window_start, train_start, fetch_end))
                train_start = train_end

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        span_index, window_start, train_start, fetch_end = self.windows[index]
        span = self.cache.spans[span_index]
        span_end = int(span["end"])
        arrays = self.cache.arrays
        length = fetch_end - window_start
        loss_mask = np.zeros(length, dtype=np.bool_)
        loss_mask[train_start - window_start : min(
            train_start - window_start + self.sequence_length,
            span_end - window_start,
        )] = True
        indices = np.arange(window_start, fetch_end)
        has_next = indices + 1 < span_end

        def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            # Copy each short slice: torch cannot safely wrap read-only mmap views,
            # and the copy avoids retaining file-page references through a batch.
            return torch.tensor(np.array(arrays[name][window_start:fetch_end]), dtype=dtype)

        return {
            "global": tensor("global", torch.float32),
            "candidates": tensor("candidates", torch.float32),
            "legal": tensor("legal", torch.bool),
            "actions": tensor("actions", torch.long),
            "rewards": tensor("rewards", torch.float32),
            "dones": tensor("dones", torch.bool),
            "transition_steps": tensor("transition_steps", torch.float32),
            "previous_actions": tensor("previous_actions", torch.long),
            "previous_rewards": tensor("previous_rewards", torch.float32),
            "outcomes": tensor("outcomes", torch.float32),
            "sequence_mask": torch.ones(length, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "has_next": torch.tensor(has_next, dtype=torch.bool),
        }


def collate_sequence_windows(rows: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    maximum = max(int(row["actions"].shape[0]) for row in rows)

    def padded(name: str, value: float | int | bool = 0) -> torch.Tensor:
        pieces = []
        for row in rows:
            source = row[name]
            padding = maximum - int(source.shape[0])
            if padding:
                shape = (padding, *source.shape[1:])
                source = torch.cat(
                    (source, torch.full(shape, value, dtype=source.dtype)), dim=0
                )
            pieces.append(source)
        return torch.stack(pieces)

    batch = {name: padded(name) for name in rows[0]}
    # The model validates that each encoded turn has a legal action. Padding is
    # masked from every loss, so give padded turns a harmless A0-only mask.
    padded_turns = ~batch["sequence_mask"]
    if bool(padded_turns.any()):
        batch["legal"][..., 0] |= padded_turns
    return batch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("A sequence batch contains no trainable turns")
    return selected.mean()


def trajectory_iql_loss(
    outputs: dict[str, torch.Tensor | None],
    target_outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    *,
    gamma: float,
    expectile: float,
    advantage_temperature: float,
    maximum_advantage_weight: float,
    behavior_clone_weight: float,
    behavior_clone_only: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs = outputs["action_log_probs"]
    q1 = outputs["q1"]
    q2 = outputs["q2"]
    values = outputs["values"]
    target_values = target_outputs["values"]
    assert isinstance(log_probs, torch.Tensor)
    assert isinstance(q1, torch.Tensor)
    assert isinstance(q2, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert isinstance(target_values, torch.Tensor)
    actions = batch["actions"]
    loss_mask = batch["loss_mask"]
    td_mask = loss_mask & (batch["dones"] | batch["has_next"])
    selected_log_probs = log_probs.gather(-1, actions[..., None]).squeeze(-1)
    selected_q1 = q1.gather(-1, actions[..., None]).squeeze(-1)
    selected_q2 = q2.gather(-1, actions[..., None]).squeeze(-1)
    if not bool(torch.isfinite(selected_log_probs[loss_mask]).all()):
        raise ValueError("A trajectory target is absent from its legal mask")

    next_values = torch.cat(
        (target_values[:, 1:], torch.zeros_like(target_values[:, :1])), dim=1
    )
    discounts = torch.pow(
        torch.full_like(batch["transition_steps"], gamma),
        batch["transition_steps"],
    )
    q_targets = batch["rewards"] + (
        (~batch["dones"]).float() * discounts * next_values.detach()
    )
    q_loss = _masked_mean(
        (selected_q1 - q_targets).square() + (selected_q2 - q_targets).square(),
        td_mask,
    )
    minimum_q = torch.minimum(selected_q1, selected_q2).detach()
    value_difference = minimum_q - values
    expectile_weights = torch.where(
        value_difference > 0,
        torch.full_like(value_difference, expectile),
        torch.full_like(value_difference, 1.0 - expectile),
    )
    value_loss = _masked_mean(expectile_weights * value_difference.square(), loss_mask)
    behavior_clone_loss = -_masked_mean(selected_log_probs, loss_mask)
    advantages = (minimum_q - values.detach()) / advantage_temperature
    advantage_weights = advantages.exp().clamp(max=maximum_advantage_weight)
    raw_selected_weights = advantage_weights[loss_mask]
    selected_weights = raw_selected_weights / raw_selected_weights.mean().clamp_min(1e-6)
    actor_loss = -(selected_weights.detach() * selected_log_probs[loss_mask]).mean()
    policy_loss = (
        behavior_clone_loss
        if behavior_clone_only
        else actor_loss + behavior_clone_weight * behavior_clone_loss
    )
    total = policy_loss + q_loss + value_loss
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs.nan_to_num()).sum(dim=-1)
    return total, {
        "total_loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "actor_loss": actor_loss.detach(),
        "behavior_clone_loss": behavior_clone_loss.detach(),
        "q_loss": q_loss.detach(),
        "value_loss": value_loss.detach(),
        "mean_advantage_weight": raw_selected_weights.mean().detach(),
        "maximum_advantage_weight_observed": raw_selected_weights.max().detach(),
        "entropy": _masked_mean(entropy, loss_mask).detach(),
        "absolute_td_error": _masked_mean(
            (minimum_q - q_targets).abs(), td_mask
        ).detach(),
    }


def _forward(
    model: TrajectoryPolicyHead,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    return model(
        batch["global"],
        batch["candidates"],
        batch["legal"],
        batch["previous_actions"],
        batch["previous_rewards"],
    )


def _evaluate(
    model: TrajectoryPolicyHead,
    target_model: TrajectoryPolicyHead,
    loader: DataLoader,
    device: torch.device,
    *,
    gamma: float,
    expectile: float,
    advantage_temperature: float,
    maximum_advantage_weight: float,
    behavior_clone_weight: float,
) -> dict[str, float]:
    model.eval()
    target_model.eval()
    totals: dict[str, float] = {}
    examples = 0
    correct = 0
    switch_targets = 0
    switch_correct = 0
    win_value_error = 0.0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            outputs = _forward(model, batch)
            target_outputs = _forward(target_model, batch)
            _loss, parts = trajectory_iql_loss(
                outputs,
                target_outputs,
                batch,
                gamma=gamma,
                expectile=expectile,
                advantage_temperature=advantage_temperature,
                maximum_advantage_weight=maximum_advantage_weight,
                behavior_clone_weight=behavior_clone_weight,
                behavior_clone_only=False,
            )
            mask = batch["loss_mask"]
            count = int(mask.sum().item())
            examples += count
            logits = outputs["policy_logits"]
            values = outputs["values"]
            assert isinstance(logits, torch.Tensor)
            assert isinstance(values, torch.Tensor)
            predictions = logits.argmax(dim=-1)
            correct += int(((predictions == batch["actions"]) & mask).sum().item())
            switch_mask = mask & (batch["actions"] >= 4) & (batch["actions"] <= 8)
            switch_targets += int(switch_mask.sum().item())
            switch_correct += int(
                ((predictions == batch["actions"]) & switch_mask).sum().item()
            )
            win_value_error += float(
                ((values - batch["outcomes"]).square() * mask).sum().item()
            )
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
    report = {key: value / max(examples, 1) for key, value in totals.items()}
    report.update(
        {
            "examples": float(examples),
            "action_accuracy": correct / max(examples, 1),
            "switch_target_accuracy": switch_correct / max(switch_targets, 1),
            "switch_targets": float(switch_targets),
            "outcome_value_mse": win_value_error / max(examples, 1),
        }
    )
    return report


def _copy_policy_checkpoint(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "interaction_head.safetensors",
        "team_preview_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "README.md",
    )
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)


def _update_target(
    target: TrajectoryPolicyHead,
    source: TrajectoryPolicyHead,
    coefficient: float,
) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.mul_(coefficient).add_(
                source_parameter, alpha=1.0 - coefficient
            )


def train_trajectory_policy(
    *,
    source_checkpoint: Path,
    train_cache: Path,
    validation_cache: Path,
    output_dir: Path,
    memory_type: str = "gru",
    epochs: int = 8,
    behavior_clone_epochs: int = 1,
    sequence_length: int = 64,
    burn_in: int = 16,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    gamma: float = 0.99,
    expectile: float = 0.7,
    advantage_temperature: float = 0.2,
    maximum_advantage_weight: float = 20.0,
    behavior_clone_weight: float = 0.1,
    target_ema: float = 0.995,
    hidden_size: int = 384,
    recurrent_layers: int = 2,
    dropout: float = 0.1,
    seed: int = 42,
    log_steps: int = 50,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if epochs <= 0 or not 0 <= behavior_clone_epochs <= epochs:
        raise ValueError("epochs must be positive and behavior_clone_epochs within epochs")
    if not 0 < gamma <= 1 or not 0.5 < expectile < 1:
        raise ValueError("gamma must be in (0,1] and expectile in (0.5,1)")
    set_seed(seed)
    train_cache_data = EncodedTrajectoryCache(train_cache)
    validation_cache_data = EncodedTrajectoryCache(validation_cache)
    signature = checkpoint_signature(source_checkpoint)
    for cache in (train_cache_data, validation_cache_data):
        if cache.metadata.get("checkpoint_signature") != signature:
            raise ValueError("Encoded trajectory cache was built from a different checkpoint")
        if not math.isclose(float(cache.metadata.get("reward_gamma", -1.0)), gamma):
            raise ValueError(
                "Training gamma must match the discount used to aggregate cached rewards"
            )
    d_model = int(train_cache_data.metadata["d_model"])
    if int(validation_cache_data.metadata["d_model"]) != d_model:
        raise ValueError("Train and validation cache dimensions differ")
    train_dataset = SequenceWindowDataset(
        train_cache_data, sequence_length=sequence_length, burn_in=burn_in
    )
    validation_dataset = SequenceWindowDataset(
        validation_cache_data, sequence_length=sequence_length, burn_in=burn_in
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence_windows,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequence_windows,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPolicyHead(
        d_model,
        memory_type=memory_type,
        hidden_size=hidden_size,
        recurrent_layers=recurrent_layers,
        dropout=dropout,
    ).to(device)
    target_model = copy.deepcopy(model).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    best_state: dict[str, torch.Tensor] | None = None
    best_target_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    best_validation_nll = math.inf
    history = []
    updates = 0
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for cpu_batch in train_loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            outputs = _forward(model, batch)
            with torch.no_grad():
                target_outputs = _forward(target_model, batch)
            loss, parts = trajectory_iql_loss(
                outputs,
                target_outputs,
                batch,
                gamma=gamma,
                expectile=expectile,
                advantage_temperature=advantage_temperature,
                maximum_advantage_weight=maximum_advantage_weight,
                behavior_clone_weight=behavior_clone_weight,
                behavior_clone_only=epoch < behavior_clone_epochs,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            _update_target(target_model, model, target_ema)
            updates += 1
            count = int(batch["loss_mask"].sum().item())
            examples += count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
            if updates == 1 or updates % log_steps == 0:
                print(
                    json.dumps(
                        {
                            "phase": "trajectory-train",
                            "memory_type": memory_type,
                            "epoch": epoch + 1,
                            "step": updates,
                            "examples": examples,
                            "learning_rate": scheduler.get_last_lr()[0],
                            **{
                                key: value / max(examples, 1)
                                for key, value in totals.items()
                            },
                        }
                    ),
                    flush=True,
                )
        validation = _evaluate(
            model,
            target_model,
            validation_loader,
            device,
            gamma=gamma,
            expectile=expectile,
            advantage_temperature=advantage_temperature,
            maximum_advantage_weight=maximum_advantage_weight,
            behavior_clone_weight=behavior_clone_weight,
        )
        epoch_row = {
            "epoch": epoch + 1,
            "behavior_clone_only": epoch < behavior_clone_epochs,
            "train": {
                key: value / max(examples, 1) for key, value in totals.items()
            },
            "validation": validation,
        }
        history.append(epoch_row)
        print(json.dumps({"phase": "trajectory-validation", **epoch_row}), flush=True)
        best_validation_nll = min(
            best_validation_nll, float(validation["behavior_clone_loss"])
        )
        # Select only from IQL epochs and use the complete held-out objective.
        # Selecting on human-action NLL would systematically undo intentional
        # advantage-weighted deviations from the behavior policy.
        selection_metric = float(validation["total_loss"])
        if epoch >= behavior_clone_epochs and selection_metric < best_validation:
            best_validation = selection_metric
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_target_state = {
                key: value.detach().cpu().clone()
                for key, value in target_model.state_dict().items()
            }
        elif best_state is None and epoch + 1 == epochs:
            # A pure behavior-cloning smoke run (epochs == BC epochs) remains
            # deployable, but normal runs always select an IQL epoch above.
            best_validation = selection_metric
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_target_state = {
                key: value.detach().cpu().clone()
                for key, value in target_model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Trajectory training produced no checkpoint")
    model.load_state_dict(best_state)
    if best_target_state is not None:
        target_model.load_state_dict(best_target_state)
    final_validation = _evaluate(
        model,
        target_model,
        validation_loader,
        device,
        gamma=gamma,
        expectile=expectile,
        advantage_temperature=advantage_temperature,
        maximum_advantage_weight=maximum_advantage_weight,
        behavior_clone_weight=behavior_clone_weight,
    )
    _copy_policy_checkpoint(source_checkpoint, output_dir)
    save_trajectory_head(model, output_dir)
    source_metadata_path = source_checkpoint / "training_config.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    report = {
        "schema": "trajectory-iql-training-v1",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_signature": signature,
        "memory_type": memory_type,
        "train_cache": str(train_cache),
        "validation_cache": str(validation_cache),
        "train_trajectories": len(train_cache_data.spans),
        "validation_trajectories": len(validation_cache_data.spans),
        "train_transitions": int(train_cache_data.metadata["rows"]),
        "validation_transitions": int(validation_cache_data.metadata["rows"]),
        "epochs": epochs,
        "behavior_clone_epochs": behavior_clone_epochs,
        "sequence_length": sequence_length,
        "burn_in": burn_in,
        "batch_size": batch_size,
        "updates": updates,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "expectile": expectile,
        "advantage_temperature": advantage_temperature,
        "maximum_advantage_weight": maximum_advantage_weight,
        "behavior_clone_weight": behavior_clone_weight,
        "target_ema": target_ema,
        "best_validation_objective": best_validation,
        "best_validation_nll": best_validation_nll,
        "final_validation": final_validation,
        "history": history,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    config = source_metadata | model.config() | report | {
        "training_objective": "trajectory-next-state-iql",
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a memoryless or recurrent candidate policy with next-state IQL."
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-type", choices=("none", "gru"), default="gru")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--behavior-clone-epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=0.2)
    parser.add_argument("--maximum-advantage-weight", type=float, default=20.0)
    parser.add_argument("--behavior-clone-weight", type=float, default=0.1)
    parser.add_argument("--target-ema", type=float, default=0.995)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_trajectory_policy(
        source_checkpoint=args.source_checkpoint,
        train_cache=args.train_cache,
        validation_cache=args.validation_cache,
        output_dir=args.output_dir,
        memory_type=args.memory_type,
        epochs=args.epochs,
        behavior_clone_epochs=args.behavior_clone_epochs,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        expectile=args.expectile,
        advantage_temperature=args.advantage_temperature,
        maximum_advantage_weight=args.maximum_advantage_weight,
        behavior_clone_weight=args.behavior_clone_weight,
        target_ema=args.target_ema,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
        dropout=args.dropout,
        seed=args.seed,
        log_steps=args.log_steps,
    )


if __name__ == "__main__":
    main()
