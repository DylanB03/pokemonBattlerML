from __future__ import annotations

import argparse
import bisect
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

from pokemon_battler.interaction_cache import (
    ARRAY_SPECS,
    INTERACTION_CACHE_SCHEMA,
)
from pokemon_battler.interaction_modeling import InteractionPolicyHead
from pokemon_battler.modeling import load_training_metadata
from pokemon_battler.reinforcement import offline_outcome_loss
from pokemon_battler.structured_modeling import (
    STRUCTURED_POLICY_SCHEMA,
    initialize_structured_head,
    save_structured_head,
)
from pokemon_battler.train import set_seed


class StructuredCacheShard(Dataset[dict[str, Any]]):
    """Read feature tensors and policy targets without reparsing prepared JSON."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        self.target_metadata = json.loads(
            (cache_dir / "structured_targets.json").read_text(encoding="utf-8")
        )
        self.rows = int(self.metadata["rows"])
        if int(self.target_metadata["rows"]) != self.rows:
            raise ValueError(f"Feature and target row counts differ in {cache_dir}")
        self._features: dict[str, np.ndarray[Any, Any]] = {}
        self._targets: dict[str, np.ndarray[Any, Any]] = {}

    def __len__(self) -> int:
        return self.rows

    def _feature(self, name: str) -> np.ndarray[Any, Any]:
        if name not in self._features:
            filename = self.metadata["arrays"][name]["file"]
            self._features[name] = np.load(self.cache_dir / filename, mmap_mode="r")
        return self._features[name]

    def _target(self, name: str) -> np.ndarray[Any, Any]:
        if name not in self._targets:
            filename = self.target_metadata["files"][name]
            self._targets[name] = np.load(self.cache_dir / filename, mmap_mode="r")
        return self._targets[name]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "_interaction_features": {name: self._feature(name)[index] for name in ARRAY_SPECS},
            "action_id": int(self._target("actions")[index]),
            "outcome_code": int(self._target("outcomes")[index]),
            "battle_decision_count": int(self._target("decision_counts")[index]),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_features"] = {}
        state["_targets"] = {}
        return state


class ShardedInteractionDataset(Dataset[dict[str, Any]]):
    """Pair each prepared JSONL part with its memory-mapped feature cache."""

    def __init__(self, prepared_dir: Path, cache_dir: Path, split: str) -> None:
        self.datasets: list[StructuredCacheShard] = []
        self.ends: list[int] = []
        source_dir = prepared_dir / split
        split_cache_dir = cache_dir / split
        if not source_dir.is_dir() or not split_cache_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} prepared data or interaction caches")
        total = 0
        for source in sorted(source_dir.glob("*.jsonl")):
            cache = split_cache_dir / f"{source.stem}.{INTERACTION_CACHE_SCHEMA}"
            dataset = StructuredCacheShard(cache)
            self.datasets.append(dataset)
            total += len(dataset)
            self.ends.append(total)
        if not self.datasets:
            raise ValueError(f"No cached {split} shards found")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        return self.datasets[shard_index][index - start]


class StructuredPolicyCollator:
    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [row["_interaction_features"] for row in rows]

        def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.stack(
                [torch.tensor(np.asarray(item[name]), dtype=dtype) for item in features]
            )

        actions = [int(row["action_id"]) for row in rows]
        outcomes = [int(row.get("outcome_code", 0)) for row in rows]
        decisions = [max(int(row.get("battle_decision_count", 1)), 1) for row in rows]
        return {
            "interaction_global_numeric": stack("global_numeric", torch.float32),
            "interaction_global_ids": stack("global_ids", torch.long),
            "interaction_pokemon_numeric": stack("pokemon_numeric", torch.float32),
            "interaction_pokemon_ids": stack("pokemon_ids", torch.long),
            "interaction_pokemon_mask": stack("pokemon_mask", torch.bool),
            "interaction_candidate_numeric": stack("candidate_numeric", torch.float32),
            "interaction_candidate_ids": stack("candidate_ids", torch.long),
            "legal_action_mask": stack("candidate_mask", torch.bool),
            "interaction_candidate_actor_slot": stack("candidate_actor_slot", torch.long),
            "interaction_history_numeric": stack("history_numeric", torch.float32),
            "interaction_history_ids": stack("history_ids", torch.long),
            "interaction_history_mask": stack("history_mask", torch.bool),
            "action_ids": torch.tensor(actions, dtype=torch.long),
            "action_family_ids": torch.tensor(
                [0 if action < 4 else 1 if action < 9 else 2 for action in actions],
                dtype=torch.long,
            ),
            "value_targets": torch.tensor(
                [1.0 if value > 0 else 0.0 if value < 0 else -1.0 for value in outcomes],
                dtype=torch.float32,
            ),
            "value_weights": torch.tensor(
                [min(max(32.0 / count, 0.25), 4.0) for count in decisions],
                dtype=torch.float32,
            ),
        }


def _forward(
    head: InteractionPolicyHead, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state_hidden = torch.zeros(
        (batch["action_ids"].shape[0], head.qwen_hidden_size),
        dtype=torch.float32,
        device=batch["action_ids"].device,
    )
    return head(state_hidden, batch)


def _loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    family_weight: float,
    expectile: float,
    advantage_temperature: float,
    maximum_advantage_weight: float,
    behavior_clone_weight: float,
    q_weight: float,
    value_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outcome_loss, outcome_parts = offline_outcome_loss(
        outputs,
        batch,
        expectile=expectile,
        advantage_temperature=advantage_temperature,
        max_advantage_weight=maximum_advantage_weight,
        behavior_clone_weight=behavior_clone_weight,
        q_weight=q_weight,
        value_weight=value_weight,
    )
    family_loss = torch.nn.functional.cross_entropy(
        outputs["family_logits"], batch["action_family_ids"]
    )
    total = outcome_loss + family_weight * family_loss
    return total, {
        "loss": total.detach(),
        **outcome_parts,
        "family_loss": family_loss.detach(),
    }


def _evaluate(
    head: InteractionPolicyHead,
    loader: DataLoader,
    device: torch.device,
    *,
    family_weight: float,
    expectile: float,
    advantage_temperature: float,
    maximum_advantage_weight: float,
    behavior_clone_weight: float,
    q_weight: float,
    value_weight: float,
) -> dict[str, float]:
    head.eval()
    totals: dict[str, float] = {}
    examples = 0
    correct = 0
    switch_examples = 0
    switch_correct = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
            outputs = _forward(head, batch)
            _, parts = _loss(
                outputs,
                batch,
                family_weight=family_weight,
                expectile=expectile,
                advantage_temperature=advantage_temperature,
                maximum_advantage_weight=maximum_advantage_weight,
                behavior_clone_weight=behavior_clone_weight,
                q_weight=q_weight,
                value_weight=value_weight,
            )
            count = int(batch["action_ids"].shape[0])
            examples += count
            predictions = outputs["action_log_probs"].argmax(dim=1)
            correct += int((predictions == batch["action_ids"]).sum().item())
            switch = (batch["action_ids"] >= 4) & (batch["action_ids"] <= 8)
            switch_examples += int(switch.sum().item())
            switch_correct += int(((predictions == batch["action_ids"]) & switch).sum().item())
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * count
    return {
        **{key: value / max(examples, 1) for key, value in totals.items()},
        "examples": float(examples),
        "action_accuracy": correct / max(examples, 1),
        "switch_accuracy": switch_correct / max(switch_examples, 1),
        "switch_examples": float(switch_examples),
    }


def _copy_source_checkpoint(source: Path, output: Path) -> None:
    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "training_config.json":
            shutil.copy2(path, output / path.name)


def train_structured_policy(
    *,
    source_checkpoint: Path,
    prepared_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    family_weight: float = 0.0,
    expectile: float = 0.7,
    advantage_temperature: float = 0.1,
    maximum_advantage_weight: float = 20.0,
    behavior_clone_weight: float = 0.1,
    q_weight: float = 1.0,
    value_weight: float = 1.0,
    blend_weight: float = 0.5,
    num_workers: int = 4,
    device_name: str = "auto",
    seed: int = 42,
    log_steps: int = 100,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if epochs <= 0 or batch_size <= 0 or eval_batch_size <= 0 or num_workers < 0:
        raise ValueError("Training sizes must be positive and num_workers non-negative")
    if (
        learning_rate <= 0
        or family_weight < 0
        or behavior_clone_weight < 0
        or q_weight < 0
        or value_weight < 0
        or blend_weight < 0
    ):
        raise ValueError("Learning rate must be positive and loss/blend weights non-negative")
    if not 0.5 < expectile < 1 or advantage_temperature <= 0:
        raise ValueError("expectile must be in (0.5, 1) and advantage temperature positive")
    if maximum_advantage_weight < 1:
        raise ValueError("maximum_advantage_weight must be at least one")
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    set_seed(seed)
    metadata = load_training_metadata(source_checkpoint)
    resolved_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(resolved_device)
    head = initialize_structured_head(source_checkpoint, metadata, device)
    train_dataset = ShardedInteractionDataset(prepared_dir, cache_dir, "train")
    validation_dataset = ShardedInteractionDataset(prepared_dir, cache_dir, "validation")
    collator = StructuredPolicyCollator()
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
                "multiprocessing_context": "spawn",
            }
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(len(train_loader) * epochs, 1)
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -math.inf
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    updates = 0
    examples = 0
    started = time.monotonic()
    for epoch in range(epochs):
        head.train()
        running: dict[str, float] = {}
        running_examples = 0
        for cpu_batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
            outputs = _forward(head, batch)
            loss, parts = _loss(
                outputs,
                batch,
                family_weight=family_weight,
                expectile=expectile,
                advantage_temperature=advantage_temperature,
                maximum_advantage_weight=maximum_advantage_weight,
                behavior_clone_weight=behavior_clone_weight,
                q_weight=q_weight,
                value_weight=value_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            count = int(batch["action_ids"].shape[0])
            updates += 1
            examples += count
            running_examples += count
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + float(value.item()) * count
            if updates == 1 or updates % log_steps == 0:
                print(
                    json.dumps(
                        {
                            "phase": "structured-policy-train",
                            "epoch": epoch + 1,
                            "step": updates,
                            "examples": examples,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            **{
                                key: value / max(running_examples, 1)
                                for key, value in running.items()
                            },
                        }
                    ),
                    flush=True,
                )
                running = {}
                running_examples = 0
        validation = _evaluate(
            head,
            validation_loader,
            device,
            family_weight=family_weight,
            expectile=expectile,
            advantage_temperature=advantage_temperature,
            maximum_advantage_weight=maximum_advantage_weight,
            behavior_clone_weight=behavior_clone_weight,
            q_weight=q_weight,
            value_weight=value_weight,
        )
        record = {"epoch": epoch + 1, **validation}
        history.append(record)
        print(json.dumps({"phase": "structured-policy-validation", **record}), flush=True)
        if validation["loss"] < best_loss or (
            validation["loss"] == best_loss and validation["action_accuracy"] > best_accuracy
        ):
            best_accuracy = validation["action_accuracy"]
            best_loss = validation["loss"]
            best_state = copy.deepcopy(head.state_dict())
    if best_state is None:
        raise RuntimeError("Structured training completed without validation")
    head.load_state_dict(best_state)
    _copy_source_checkpoint(source_checkpoint, output_dir)
    save_structured_head(head, output_dir)
    report = {
        "schema": "structured-policy-training-v1",
        "source_checkpoint": str(source_checkpoint),
        "prepared_dir": str(prepared_dir),
        "cache_dir": str(cache_dir),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "epochs": epochs,
        "updates": updates,
        "examples_seen": examples,
        "learning_rate": learning_rate,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "lr_scheduler": "cosine",
        "scheduler_updates": updates,
        "warmup_ratio": 0.0,
        "weight_decay": weight_decay,
        "qwen_frozen": True,
        "qwen_learning_rate": 0.0,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "blend_weight": blend_weight,
        "expectile": expectile,
        "advantage_temperature": advantage_temperature,
        "maximum_advantage_weight": maximum_advantage_weight,
        "behavior_clone_weight": behavior_clone_weight,
        "q_weight": q_weight,
        "value_weight": value_weight,
        "workers": num_workers,
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "history": history,
    }
    combined_metadata = (
        metadata
        | report
        | {
            "structured_policy_schema": STRUCTURED_POLICY_SCHEMA,
            "structured_policy_blend_weight": blend_weight,
            "training_objective": "large-offline-structured-sidecar",
        }
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(combined_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "structured_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fast structured sidecar and retain Qwen as the base policy."
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--family-weight", type=float, default=0.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=0.1)
    parser.add_argument("--maximum-advantage-weight", type=float, default=20.0)
    parser.add_argument("--behavior-clone-weight", type=float, default=0.1)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", dest="device_name", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = train_structured_policy(
        source_checkpoint=args.source_checkpoint,
        prepared_dir=args.prepared_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        family_weight=args.family_weight,
        expectile=args.expectile,
        advantage_temperature=args.advantage_temperature,
        maximum_advantage_weight=args.maximum_advantage_weight,
        behavior_clone_weight=args.behavior_clone_weight,
        q_weight=args.q_weight,
        value_weight=args.value_weight,
        blend_weight=args.blend_weight,
        num_workers=args.num_workers,
        device_name=args.device_name,
        seed=args.seed,
        log_steps=args.log_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
