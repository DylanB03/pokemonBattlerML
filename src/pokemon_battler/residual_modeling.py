from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.interaction_modeling import (
    INTERACTION_HEAD_FILENAME,
    InteractionPolicyHead,
)

RESIDUAL_HEAD_FILENAME = "residual_head.safetensors"
RESIDUAL_CONFIG_FILENAME = "residual_config.json"
RESIDUAL_POLICY_SCHEMA = "champion-residual-policy-v1"


class ChampionPolicyScorer(torch.nn.Module):
    """Reconstruct the frozen champion policy from cached interaction embeddings."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.candidate_scorer = torch.nn.Linear(d_model, 1)
        self.family_scorer = torch.nn.Linear(d_model, 3)

    def forward(
        self,
        global_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate_scores = self.candidate_scorer(candidate_embeddings.float()).squeeze(-1)
        family_logits = self.family_scorer(global_embedding.float())
        action_log_probs, _ = InteractionPolicyHead._hierarchical_log_probabilities(
            candidate_scores, family_logits, legal_mask.bool()
        )
        return action_log_probs


class ResidualPolicyHead(torch.nn.Module):
    """A bounded correction applied on top of a frozen champion distribution.

    The final projection is initialized to exactly zero.  Consequently a newly
    created residual head preserves the champion's ordering and probabilities
    to floating-point precision instead of replacing them with a random actor.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_size: int = 256,
        dropout: float = 0.1,
        maximum_logit_delta: float = 1.5,
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_size <= 0:
            raise ValueError("Residual dimensions must be positive")
        if maximum_logit_delta <= 0:
            raise ValueError("maximum_logit_delta must be positive")
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.maximum_logit_delta = maximum_logit_delta
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model * 3 + 1),
            torch.nn.Linear(d_model * 3 + 1, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
        )
        final = self.network[-1]
        assert isinstance(final, torch.nn.Linear)
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def config(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_POLICY_SCHEMA,
            "d_model": self.d_model,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "maximum_logit_delta": self.maximum_logit_delta,
        }

    def forward(
        self,
        global_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        legal_mask: torch.Tensor,
        champion_log_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if candidate_embeddings.shape[-2] != ACTION_COUNT:
            raise ValueError("Residual policy requires exactly 13 candidate slots")
        legal = legal_mask.bool()
        if not bool(legal.any(dim=-1).all()):
            raise ValueError("Every residual-policy row needs a legal action")
        expanded_global = global_embedding.unsqueeze(-2).expand_as(candidate_embeddings)
        champion_feature = torch.where(
            legal, champion_log_probs.float(), torch.zeros_like(champion_log_probs.float())
        ).unsqueeze(-1)
        features = torch.cat(
            (
                candidate_embeddings.float(),
                expanded_global.float(),
                candidate_embeddings.float() * expanded_global.float(),
                champion_feature,
            ),
            dim=-1,
        )
        raw_delta = self.network(features).squeeze(-1)
        delta = self.maximum_logit_delta * torch.tanh(raw_delta)
        corrected = champion_log_probs.float() + delta
        corrected = corrected.masked_fill(~legal, float("-inf"))
        action_log_probs = torch.log_softmax(corrected, dim=-1)
        return {
            "action_log_probs": action_log_probs,
            "logit_deltas": delta.masked_fill(~legal, 0.0),
        }


def load_champion_scorer(
    checkpoint: str | Path,
    *,
    d_model: int,
    device: torch.device,
) -> ChampionPolicyScorer:
    path = Path(checkpoint) / INTERACTION_HEAD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    source = load_safetensors(path, device=str(device))
    keys = {
        "candidate_scorer.weight",
        "candidate_scorer.bias",
        "family_scorer.weight",
        "family_scorer.bias",
    }
    missing = keys - source.keys()
    if missing:
        raise ValueError(f"Champion policy scorer is missing tensors: {sorted(missing)}")
    scorer = ChampionPolicyScorer(d_model).to(device=device, dtype=torch.float32)
    scorer.load_state_dict(
        {
            "candidate_scorer.weight": source["candidate_scorer.weight"].float(),
            "candidate_scorer.bias": source["candidate_scorer.bias"].float(),
            "family_scorer.weight": source["family_scorer.weight"].float(),
            "family_scorer.bias": source["family_scorer.bias"].float(),
        }
    )
    scorer.requires_grad_(False)
    scorer.eval()
    return scorer


def save_residual_head(head: ResidualPolicyHead, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().float().contiguous()
        for key, value in head.state_dict().items()
    }
    save_safetensors(state, destination / RESIDUAL_HEAD_FILENAME)
    (destination / RESIDUAL_CONFIG_FILENAME).write_text(
        json.dumps(head.config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def has_residual_head(checkpoint: str | Path | None) -> bool:
    return bool(checkpoint) and (
        Path(checkpoint) / RESIDUAL_HEAD_FILENAME
    ).is_file() and (Path(checkpoint) / RESIDUAL_CONFIG_FILENAME).is_file()


def load_residual_head(
    checkpoint: str | Path,
    device: torch.device,
) -> ResidualPolicyHead:
    root = Path(checkpoint)
    config = json.loads((root / RESIDUAL_CONFIG_FILENAME).read_text(encoding="utf-8"))
    if config.get("schema") != RESIDUAL_POLICY_SCHEMA:
        raise ValueError("Residual policy schema does not match this code")
    head = ResidualPolicyHead(
        int(config["d_model"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        maximum_logit_delta=float(config["maximum_logit_delta"]),
    ).to(device=device, dtype=torch.float32)
    head.load_state_dict(
        load_safetensors(root / RESIDUAL_HEAD_FILENAME, device=str(device))
    )
    return head
