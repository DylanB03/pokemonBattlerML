from __future__ import annotations

import argparse
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

from pokemon_battler.core.actions import ACTION_COUNT, legal_action_ids
from pokemon_battler.evaluation.evaluation_utils import ActionMetrics, select_evaluation_dataset
from pokemon_battler.core.mechanics_v2 import (
    MECHANICS_FEATURE_COUNT,
    MECHANICS_FEATURE_NAMES,
    MECHANICS_IDENTITY_FIELDS,
    MECHANICS_IDENTITY_NAMES,
    MECHANICS_IDENTITY_VOCAB_SIZES,
    MECHANICS_SCHEMA,
    candidate_feature_matrix,
    candidate_identity_matrix,
)
from pokemon_battler.data.mechanics_cache import build_feature_cache, default_cache_path
from pokemon_battler.data.training_data import (
    JsonlOffsetDataset,
    MechanicsCacheDataset,
    state_with_row_context,
)

WEIGHTS_FILENAME = "mechanics_baseline.safetensors"
CONFIG_FILENAME = "mechanics_baseline.json"


class MechanicsOnlyPolicy(torch.nn.Module):
    """A small shared scorer that receives no language-model representation."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.identity_embedding_size = max(min(hidden_size // 8, 16), 8)
        namespaces = {namespace for _, namespace in MECHANICS_IDENTITY_FIELDS}
        self.identity_embeddings = torch.nn.ModuleDict(
            {
                f"namespace_{namespace}": torch.nn.Embedding(
                    MECHANICS_IDENTITY_VOCAB_SIZES[namespace],
                    self.identity_embedding_size,
                    padding_idx=0,
                )
                for namespace in sorted(namespaces)
            }
        )
        scorer_input_size = MECHANICS_FEATURE_COUNT + (
            len(MECHANICS_IDENTITY_FIELDS) * self.identity_embedding_size
        )
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(scorer_input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )
        for embedding in self.identity_embeddings.values():
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(embedding.weight[0])

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        identity_ids = batch["mechanics_identity_ids"]
        identities = [
            self.identity_embeddings[f"namespace_{namespace}"](
                identity_ids[:, :, index]
            )
            for index, (_, namespace) in enumerate(MECHANICS_IDENTITY_FIELDS)
        ]
        inputs = torch.cat(
            (batch["mechanics_features"].float(), *identities),
            dim=-1,
        )
        logits = self.scorer(inputs).squeeze(-1)
        return logits.masked_fill(~batch["legal_action_mask"], float("-inf"))


class MechanicsOnlyCollator:
    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        features: list[torch.Tensor] = []
        identities: list[torch.Tensor] = []
        legal_masks: list[list[bool]] = []
        targets: list[int] = []
        for row in rows:
            state = state_with_row_context(row)
            legal = legal_action_ids(state)
            target = int(row["action_id"])
            if target not in legal:
                raise ValueError(f"Target A{target} is absent from the legal-action mask")
            raw_features = row.get("_mechanics_features")
            if raw_features is None:
                raw_features = candidate_feature_matrix(state)
            feature_tensor = torch.as_tensor(raw_features, dtype=torch.float32)
            if tuple(feature_tensor.shape) != (ACTION_COUNT, MECHANICS_FEATURE_COUNT):
                raise ValueError("Invalid candidate mechanics matrix")
            features.append(feature_tensor)
            identities.append(
                torch.tensor(candidate_identity_matrix(state), dtype=torch.long)
            )
            legal_masks.append([action_id in legal for action_id in range(ACTION_COUNT)])
            targets.append(target)
        return {
            "mechanics_features": torch.stack(features),
            "mechanics_identity_ids": torch.stack(identities),
            "legal_action_mask": torch.tensor(legal_masks, dtype=torch.bool),
            "action_ids": torch.tensor(targets, dtype=torch.long),
            "rows": list(rows),
        }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def _evaluate_loader(
    model: MechanicsOnlyPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch)
        targets = batch["action_ids"]
        total_loss += float(
            torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        )
        total += int(targets.numel())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
    model.train()
    return {
        "validation_loss": total_loss / total if total else math.nan,
        "validation_accuracy": correct / total if total else math.nan,
    }


def _save(model: MechanicsOnlyPolicy, output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        output_dir / WEIGHTS_FILENAME,
    )
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: str | Path, device: torch.device) -> tuple[MechanicsOnlyPolicy, dict[str, Any]]:
    root = Path(path)
    config = json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))
    if config.get("mechanics_schema") != MECHANICS_SCHEMA:
        raise ValueError("Mechanics baseline checkpoint uses an incompatible feature schema")
    model = MechanicsOnlyPolicy(int(config["hidden_size"])).to(device)
    model.load_state_dict(load_safetensors(root / WEIGHTS_FILENAME, device=str(device)))
    return model, config


def _cached_dataset(data_file: str, cache_file: str | None, rebuild: bool) -> Dataset:
    path = Path(cache_file) if cache_file else default_cache_path(data_file, MECHANICS_SCHEMA)
    build_feature_cache(data_file, path, schema=MECHANICS_SCHEMA, overwrite=rebuild)
    return MechanicsCacheDataset(
        JsonlOffsetDataset(data_file),
        path,
        mechanics_schema=MECHANICS_SCHEMA,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_size <= 0:
        raise ValueError("epochs, batch-size, and hidden-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complete_train = _cached_dataset(args.train_file, args.train_cache, args.rebuild_cache)
    train_dataset, train_sample = select_evaluation_dataset(
        complete_train,
        max_examples=args.max_train_examples,
        mode=args.sample_mode,
        seed=args.seed,
    )
    validation_dataset: Dataset | None = None
    validation_sample: dict[str, Any] | None = None
    if args.validation_file:
        complete_validation = _cached_dataset(
            args.validation_file,
            args.validation_cache,
            args.rebuild_cache,
        )
        validation_dataset, validation_sample = select_evaluation_dataset(
            complete_validation,
            max_examples=args.max_validation_examples,
            mode=args.sample_mode,
            seed=args.seed,
        )
    collator = MechanicsOnlyCollator()
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
    model = MechanicsOnlyPolicy(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite-output to reuse it."
        )
    started = time.monotonic()
    best_accuracy = -math.inf
    best_loss = math.inf
    global_step = 0
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, batch["action_ids"])
            loss.backward()
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
            if validation_loader is not None and global_step % args.eval_steps == 0:
                metrics = _evaluate_loader(model, validation_loader, device)
                record = {"step": global_step, **metrics}
                print(json.dumps(record), flush=True)
                history.append(record)
                accuracy = metrics["validation_accuracy"]
                validation_loss = metrics["validation_loss"]
                if accuracy > best_accuracy or (
                    accuracy == best_accuracy and validation_loss < best_loss
                ):
                    best_accuracy = accuracy
                    best_loss = validation_loss
                    _save(
                        model,
                        output_dir / "best",
                        vars(args)
                        | {
                            "global_step": global_step,
                            "best_validation_accuracy": best_accuracy,
                            "best_validation_loss": best_loss,
                            "mechanics_schema": MECHANICS_SCHEMA,
                            "feature_names": list(MECHANICS_FEATURE_NAMES),
                            "identity_names": list(MECHANICS_IDENTITY_NAMES),
                            "identity_vocab_sizes": dict(
                                MECHANICS_IDENTITY_VOCAB_SIZES
                            ),
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
        "best_validation_accuracy": best_accuracy if math.isfinite(best_accuracy) else None,
        "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
        "mechanics_schema": MECHANICS_SCHEMA,
        "feature_names": list(MECHANICS_FEATURE_NAMES),
        "identity_names": list(MECHANICS_IDENTITY_NAMES),
        "identity_vocab_sizes": dict(MECHANICS_IDENTITY_VOCAB_SIZES),
        "train_sample": train_sample,
        "validation_sample": validation_sample,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    _save(model, output_dir / "final", final_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return final_config


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load(args.checkpoint, device)
    model.eval()
    complete = _cached_dataset(args.data_file, args.cache, False)
    dataset, sample = select_evaluation_dataset(
        complete,
        max_examples=args.max_examples,
        mode=args.sample_mode,
        seed=args.sample_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MechanicsOnlyCollator(),
        num_workers=args.num_workers,
    )
    metrics = ActionMetrics(max_saved_errors=args.max_saved_errors)
    evaluated = 0
    for batch in loader:
        rows = batch["rows"]
        batch = _move_batch(batch, device)
        logits = model(batch).float().cpu()
        masks = batch["legal_action_mask"].cpu()
        for row_index, row in enumerate(rows):
            legal = torch.nonzero(masks[row_index], as_tuple=False).flatten().tolist()
            scores = {action_id: float(logits[row_index, action_id]) for action_id in legal}
            metrics.add(evaluated, row, scores)
            evaluated += 1
    report = metrics.report()
    report["evaluation"] = {
        "model_type": "mechanics_only_policy",
        "checkpoint": args.checkpoint,
        "data_file": args.data_file,
        "sample": sample,
        "training_config": config,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the mechanics-only policy.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-file", required=True)
    train_parser.add_argument("--validation-file")
    train_parser.add_argument("--train-cache")
    train_parser.add_argument("--validation-cache")
    train_parser.add_argument("--rebuild-cache", action="store_true")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--overwrite-output", action="store_true")
    train_parser.add_argument("--max-train-examples", type=int)
    train_parser.add_argument("--max-validation-examples", type=int, default=5_000)
    train_parser.add_argument("--sample-mode", choices=("head", "hash"), default="hash")
    train_parser.add_argument("--hidden-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--eval-batch-size", type=int, default=512)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--eval-steps", type=int, default=250)
    train_parser.add_argument("--log-steps", type=int, default=50)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data-file", required=True)
    evaluate_parser.add_argument("--cache")
    evaluate_parser.add_argument("--max-examples", type=int, default=5_000)
    evaluate_parser.add_argument("--sample-mode", choices=("head", "hash"), default="hash")
    evaluate_parser.add_argument("--sample-seed", type=int, default=42)
    evaluate_parser.add_argument("--batch-size", type=int, default=512)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--max-saved-errors", type=int, default=100)
    evaluate_parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = train(args) if args.command == "train" else evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
