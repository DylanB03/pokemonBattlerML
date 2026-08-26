from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader, Dataset

from pokemon_battler.core.actions import (
    ACTION_COUNT,
    describe_action,
    legal_action_ids,
    sorted_moves,
    sorted_switches,
)
from pokemon_battler.evaluation.evaluation_utils import (
    ActionMetrics,
    load_action_counts,
    select_evaluation_dataset,
)
from pokemon_battler.data.training_data import JsonlOffsetDataset

BASELINE_WEIGHTS = "baseline_model.safetensors"
BASELINE_CONFIG = "baseline_config.json"

STATE_FIELDS = (
    "format",
    "forced_switch",
    "can_tera",
    "weather",
    "battle_field",
    "player_conditions",
    "opponent_conditions",
    "opponents_remaining",
)
POKEMON_FIELDS = (
    "name",
    "base_species",
    "types",
    "tera_type",
    "item",
    "ability",
    "status",
    "effect",
)
BOOST_FIELDS = (
    "atk_boost",
    "def_boost",
    "spa_boost",
    "spd_boost",
    "spe_boost",
    "accuracy_boost",
    "evasion_boost",
)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(_scalar(item) for item in value)
    if value is None:
        return "unknown"
    return str(value).lower().replace(" ", "")


def _hp_bucket(value: Any) -> str:
    try:
        return str(max(0, min(10, int(float(value) * 10))))
    except (TypeError, ValueError):
        return "unknown"


def _previous_move(value: Any) -> str:
    if isinstance(value, dict):
        return _scalar(value.get("name", "nomove"))
    return _scalar(value or "nomove")


def _pokemon_features(pokemon: dict[str, Any], role: str) -> list[str]:
    features = [
        f"{role}.{field}={_scalar(pokemon.get(field, 'unknown'))}"
        for field in POKEMON_FIELDS
    ]
    features.append(f"{role}.hp_bucket={_hp_bucket(pokemon.get('hp_pct'))}")
    features.extend(
        f"{role}.{field}={_scalar(pokemon.get(field, 0))}" for field in BOOST_FIELDS
    )
    for move_index, move in enumerate(sorted_moves(pokemon)):
        move_name = _scalar(move.get("name", "unknown"))
        features.extend(
            (
                f"{role}.move{move_index}.name={move_name}",
                f"{role}.move_name={move_name}",
                f"{role}.move{move_index}.type={_scalar(move.get('move_type', 'unknown'))}",
                f"{role}.move{move_index}.category={_scalar(move.get('category', 'unknown'))}",
            )
        )
    return features


def global_features(state: dict[str, Any]) -> list[str]:
    features = ["global_bias=1"]
    features.extend(
        f"state.{field}={_scalar(state.get(field, 'unknown'))}" for field in STATE_FIELDS
    )
    features.extend(_pokemon_features(state["player_active_pokemon"], "player_active"))
    features.extend(_pokemon_features(state["opponent_active_pokemon"], "opponent_active"))
    features.extend(
        (
            f"state.player_prev_move={_previous_move(state.get('player_prev_move'))}",
            f"state.opponent_prev_move={_previous_move(state.get('opponent_prev_move'))}",
        )
    )
    for species in state.get("opponent_teampreview") or []:
        features.append(f"opponent_preview.species={_scalar(species)}")
    for switch_index, pokemon in enumerate(sorted_switches(state)):
        species = _scalar(pokemon.get("name", "unknown"))
        features.extend(
            (
                f"available_switch{switch_index}.species={species}",
                f"available_switch.species={species}",
                f"available_switch{switch_index}.hp_bucket={_hp_bucket(pokemon.get('hp_pct'))}",
                (
                    f"available_switch{switch_index}.status="
                    f"{_scalar(pokemon.get('status', 'unknown'))}"
                ),
            )
        )
    return features


def action_features(state: dict[str, Any], action_id: int) -> list[str]:
    details = describe_action(state, action_id)
    features = [
        "action_bias=1",
        f"action.id=A{action_id}",
        f"action.type={_scalar(details['type'])}",
    ]
    if details["type"] == "switch":
        pokemon = sorted_switches(state)[action_id - 4]
        features.extend(_pokemon_features(pokemon, "action_switch"))
    else:
        tera = bool(details.get("terastallize", False))
        move_index = action_id - 9 if tera else action_id
        move = sorted_moves(state["player_active_pokemon"])[move_index]
        features.extend(
            (
                f"action.name={_scalar(move.get('name', 'unknown'))}",
                f"action.move_type={_scalar(move.get('move_type', 'unknown'))}",
                f"action.category={_scalar(move.get('category', 'unknown'))}",
                f"action.base_power={_scalar(move.get('base_power', 0))}",
                f"action.priority={_scalar(move.get('priority', 0))}",
                f"action.tera={_scalar(tera)}",
            )
        )
    return features


def _feature_id(value: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_buckets


class FeatureCollator:
    def __init__(self, num_buckets: int) -> None:
        self.num_buckets = num_buckets

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        global_rows: list[list[int]] = []
        action_rows: list[list[list[int]]] = []
        legal_rows: list[list[bool]] = []
        targets: list[int] = []
        for row in rows:
            state = row["state"]
            legal = legal_action_ids(state)
            target = int(row["action_id"])
            if target not in legal:
                raise ValueError(f"Target A{target} is absent from the legal-action mask")
            global_rows.append(
                [_feature_id(value, self.num_buckets) for value in global_features(state)]
            )
            per_action: list[list[int]] = []
            for action_id in range(ACTION_COUNT):
                values = action_features(state, action_id) if action_id in legal else ["illegal"]
                per_action.append(
                    [_feature_id(value, self.num_buckets) for value in values]
                )
            action_rows.append(per_action)
            legal_rows.append([action_id in legal for action_id in range(ACTION_COUNT)])
            targets.append(target)

        max_global = max(len(values) for values in global_rows)
        max_action = max(
            len(values) for per_action in action_rows for values in per_action
        )
        global_ids = torch.zeros((len(rows), max_global), dtype=torch.long)
        global_mask = torch.zeros((len(rows), max_global), dtype=torch.bool)
        action_ids = torch.zeros(
            (len(rows), ACTION_COUNT, max_action), dtype=torch.long
        )
        action_mask = torch.zeros(
            (len(rows), ACTION_COUNT, max_action), dtype=torch.bool
        )
        for row_index, values in enumerate(global_rows):
            global_ids[row_index, : len(values)] = torch.tensor(values)
            global_mask[row_index, : len(values)] = True
            for action_id, action_values in enumerate(action_rows[row_index]):
                action_ids[row_index, action_id, : len(action_values)] = torch.tensor(
                    action_values
                )
                action_mask[row_index, action_id, : len(action_values)] = True
        return {
            "global_feature_ids": global_ids,
            "global_feature_mask": global_mask,
            "action_feature_ids": action_ids,
            "action_feature_mask": action_mask,
            "legal_action_mask": torch.tensor(legal_rows, dtype=torch.bool),
            "action_ids": torch.tensor(targets, dtype=torch.long),
            "rows": list(rows),
        }


class HashedActionPolicy(torch.nn.Module):
    """Small feature-hashed action scorer used as a non-language-model baseline."""

    def __init__(self, num_buckets: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(num_buckets, embedding_dim)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        dimension: int,
    ) -> torch.Tensor:
        weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        total = (embeddings * weights).sum(dim=dimension)
        denominator = weights.sum(dim=dimension).clamp_min(1.0)
        return total / denominator

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        global_embeddings = self.embedding(batch["global_feature_ids"])
        global_state = self._masked_mean(
            global_embeddings,
            batch["global_feature_mask"],
            dimension=1,
        )
        action_embeddings = self.embedding(batch["action_feature_ids"])
        actions = self._masked_mean(
            action_embeddings,
            batch["action_feature_mask"],
            dimension=2,
        )
        expanded_global = global_state[:, None, :].expand(-1, ACTION_COUNT, -1)
        combined = torch.cat(
            (
                expanded_global,
                actions,
                expanded_global * actions,
                torch.abs(expanded_global - actions),
            ),
            dim=-1,
        )
        logits = self.scorer(combined).squeeze(-1)
        return logits.masked_fill(~batch["legal_action_mask"], float("-inf"))


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def _validation_loss(
    model: HashedActionPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(batch, device)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch["action_ids"], reduction="sum")
        total_loss += float(loss.item())
        total_examples += int(batch["action_ids"].numel())
    model.train()
    return total_loss / total_examples if total_examples else math.nan


def _save_baseline(
    model: HashedActionPolicy,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
    }
    save_safetensors(weights, output_dir / BASELINE_WEIGHTS)
    (output_dir / BASELINE_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_baseline(
    path: str | Path,
    device: torch.device,
) -> tuple[HashedActionPolicy, dict[str, Any]]:
    root = Path(path)
    config = json.loads((root / BASELINE_CONFIG).read_text(encoding="utf-8"))
    model = HashedActionPolicy(
        int(config["num_buckets"]),
        int(config["embedding_dim"]),
        int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(load_safetensors(root / BASELINE_WEIGHTS, device=str(device)))
    return model, config


def train_baseline(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_buckets <= 0:
        raise ValueError("epochs, batch_size, and num_buckets must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = JsonlOffsetDataset(args.train_file)
    train_dataset, train_sample = select_evaluation_dataset(
        full_train,
        max_examples=args.max_train_examples,
        mode=args.sample_mode,
        seed=args.seed,
    )
    validation_dataset: Dataset[dict[str, Any]] | None = None
    validation_sample: dict[str, Any] | None = None
    if args.validation_file:
        full_validation = JsonlOffsetDataset(args.validation_file)
        validation_dataset, validation_sample = select_evaluation_dataset(
            full_validation,
            max_examples=args.max_validation_examples,
            mode=args.sample_mode,
            seed=args.seed,
        )

    collator = FeatureCollator(args.num_buckets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )
    model = HashedActionPolicy(
        args.num_buckets,
        args.embedding_dim,
        args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite-output to reuse it."
        )
    started = time.monotonic()
    global_step = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, batch["action_ids"])
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % args.log_steps == 0:
                record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": float(loss.item()),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
                print(json.dumps(record), flush=True)
                history.append(record)

            if (
                validation_loader is not None
                and args.eval_steps > 0
                and global_step % args.eval_steps == 0
            ):
                value = _validation_loss(
                    model,
                    validation_loader,
                    device,
                    args.eval_batches,
                )
                record = {"step": global_step, "validation_loss": value}
                print(json.dumps(record), flush=True)
                history.append(record)
                if value < best_validation_loss:
                    best_validation_loss = value
                    _save_baseline(
                        model,
                        output_dir / "best",
                        vars(args)
                        | {
                            "global_step": global_step,
                            "best_validation_loss": value,
                            "train_sample": train_sample,
                            "validation_sample": validation_sample,
                        },
                    )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final_config = vars(args) | {
        "global_step": global_step,
        "best_validation_loss": (
            best_validation_loss if math.isfinite(best_validation_loss) else None
        ),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "train_sample": train_sample,
        "validation_sample": validation_sample,
    }
    _save_baseline(model, output_dir / "final", final_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_config


@torch.inference_mode()
def evaluate_baseline(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_baseline(args.checkpoint, device)
    model.eval()
    full_dataset = JsonlOffsetDataset(args.data_file)
    dataset, sample = select_evaluation_dataset(
        full_dataset,
        max_examples=args.max_examples,
        mode=args.sample_mode,
        seed=args.sample_seed,
    )
    collator = FeatureCollator(model.num_buckets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_counts = (
        load_action_counts(args.baseline_train_file) if args.baseline_train_file else None
    )
    metrics = ActionMetrics(
        train_action_counts=train_counts,
        max_saved_errors=args.max_saved_errors,
    )
    evaluated = 0
    for batch in loader:
        rows = batch["rows"]
        batch = _move_batch(batch, device)
        logits = model(batch).float().cpu()
        legal_masks = batch["legal_action_mask"].cpu()
        for row_index, row in enumerate(rows):
            legal = torch.nonzero(legal_masks[row_index], as_tuple=False).flatten().tolist()
            scores = {action_id: float(logits[row_index, action_id]) for action_id in legal}
            metrics.add(evaluated, row, scores)
            evaluated += 1
        if args.log_every and evaluated % args.log_every < len(rows):
            print(
                json.dumps(
                    {"evaluated": evaluated, "accuracy": metrics.correct / evaluated}
                ),
                flush=True,
            )
    report = metrics.report()
    report["evaluation"] = {
        "model_type": "hashed_action_policy",
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "sample": sample,
        "training_config": config,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def evaluate_static_baselines(args: argparse.Namespace) -> dict[str, Any]:
    """Measure uniform and training-frequency baselines without loading a model."""
    full_dataset = JsonlOffsetDataset(args.data_file)
    dataset, sample = select_evaluation_dataset(
        full_dataset,
        max_examples=args.max_examples,
        mode=args.sample_mode,
        seed=args.sample_seed,
    )
    train_counts = (
        load_action_counts(args.baseline_train_file) if args.baseline_train_file else None
    )
    metrics = ActionMetrics(train_action_counts=train_counts, max_saved_errors=0)
    for index, row in enumerate(dataset):
        legal = [int(value) for value in row["legal_action_ids"]]
        if train_counts is None:
            scores = {action_id: 0.0 for action_id in legal}
        else:
            scores = {action_id: float(train_counts[action_id]) for action_id in legal}
        metrics.add(index, row, scores)
    complete_report = metrics.report()
    report = {
        "examples": complete_report["examples"],
        "baselines": complete_report["baselines"],
        "target_counts": complete_report["target_counts"],
        "sample_coverage": complete_report["sample_coverage"],
        "evaluation": {
            "model_type": "static_baselines",
            "data_file": args.data_file,
            "baseline_train_file": args.baseline_train_file,
            "sample": sample,
        },
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate a compact non-language-model action baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-file", required=True)
    train_parser.add_argument("--validation-file")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--overwrite-output", action="store_true")
    train_parser.add_argument("--max-train-examples", type=int, default=100_000)
    train_parser.add_argument("--max-validation-examples", type=int, default=5_000)
    train_parser.add_argument("--sample-mode", choices=("head", "hash"), default="hash")
    train_parser.add_argument("--num-buckets", type=int, default=65_536)
    train_parser.add_argument("--embedding-dim", type=int, default=32)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--eval-batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--eval-steps", type=int, default=250)
    train_parser.add_argument("--eval-batches", type=int)
    train_parser.add_argument("--log-steps", type=int, default=50)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data-file", required=True)
    evaluate_parser.add_argument("--max-examples", type=int, default=5_000)
    evaluate_parser.add_argument("--sample-mode", choices=("head", "hash"), default="hash")
    evaluate_parser.add_argument("--sample-seed", type=int, default=42)
    evaluate_parser.add_argument("--baseline-train-file")
    evaluate_parser.add_argument("--batch-size", type=int, default=256)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--max-saved-errors", type=int, default=100)
    evaluate_parser.add_argument("--log-every", type=int, default=1000)
    evaluate_parser.add_argument("--output")

    static_parser = subparsers.add_parser("static")
    static_parser.add_argument("--data-file", required=True)
    static_parser.add_argument("--baseline-train-file")
    static_parser.add_argument("--max-examples", type=int, default=5_000)
    static_parser.add_argument("--sample-mode", choices=("head", "hash"), default="hash")
    static_parser.add_argument("--sample-seed", type=int, default=42)
    static_parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = train_baseline(args)
    elif args.command == "static":
        result = evaluate_static_baselines(args)
    else:
        result = evaluate_baseline(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
