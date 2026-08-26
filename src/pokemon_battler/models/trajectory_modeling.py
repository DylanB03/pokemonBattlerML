from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from pokemon_battler.core.actions import ACTION_COUNT
from pokemon_battler.data.trajectory_prepare import PREVIOUS_ACTION_SENTINEL

TRAJECTORY_HEAD_FILENAME = "trajectory_head.safetensors"
TRAJECTORY_MODEL_SCHEMA = "recurrent-candidate-iql-v1"


class TrajectoryPolicyHead(torch.nn.Module):
    """Score each legal candidate after carrying learned state across turns."""

    def __init__(
        self,
        d_model: int,
        *,
        memory_type: str = "gru",
        hidden_size: int = 384,
        recurrent_layers: int = 2,
        action_embedding_size: int = 32,
        reward_embedding_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if memory_type not in {"none", "gru"}:
            raise ValueError("memory_type must be 'none' or 'gru'")
        self.schema = TRAJECTORY_MODEL_SCHEMA
        self.d_model = d_model
        self.memory_type = memory_type
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.action_embedding_size = action_embedding_size
        self.reward_embedding_size = reward_embedding_size
        self.dropout = dropout

        self.previous_action_embedding = torch.nn.Embedding(
            PREVIOUS_ACTION_SENTINEL + 1, action_embedding_size
        )
        self.previous_reward_projection = torch.nn.Sequential(
            torch.nn.Linear(1, reward_embedding_size),
            torch.nn.Tanh(),
        )
        input_size = d_model + action_embedding_size + reward_embedding_size
        self.input_norm = torch.nn.LayerNorm(input_size)
        if memory_type == "gru":
            self.memory: torch.nn.Module = torch.nn.GRU(
                input_size,
                hidden_size,
                num_layers=recurrent_layers,
                dropout=dropout if recurrent_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.memory = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
            )
        self.candidate_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, hidden_size),
        )
        self.policy_scorer = self._candidate_scorer()
        self.q1_scorer = self._candidate_scorer()
        self.q2_scorer = self._candidate_scorer()
        self.value_scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, 1),
        )

    def _candidate_scorer(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size // 2, 1),
        )

    def config(self) -> dict[str, Any]:
        return {
            "trajectory_model_schema": self.schema,
            "trajectory_d_model": self.d_model,
            "trajectory_memory_type": self.memory_type,
            "trajectory_hidden_size": self.hidden_size,
            "trajectory_recurrent_layers": self.recurrent_layers,
            "trajectory_action_embedding_size": self.action_embedding_size,
            "trajectory_reward_embedding_size": self.reward_embedding_size,
            "trajectory_dropout": self.dropout,
        }

    def forward(
        self,
        global_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        legal_mask: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if global_embeddings.ndim != 3:
            raise ValueError("global_embeddings must have shape [batch, turns, d_model]")
        if candidate_embeddings.shape[:3] != (
            global_embeddings.shape[0],
            global_embeddings.shape[1],
            ACTION_COUNT,
        ):
            raise ValueError("candidate_embeddings must have shape [batch, turns, 13, d]")
        if not bool(legal_mask.any(dim=-1).all()):
            raise ValueError("Every unpadded trajectory turn needs at least one legal action")
        previous_actions = previous_actions.clamp(0, PREVIOUS_ACTION_SENTINEL)
        action_context = self.previous_action_embedding(previous_actions)
        reward_context = self.previous_reward_projection(previous_rewards[..., None].float())
        inputs = self.input_norm(
            torch.cat((global_embeddings.float(), action_context, reward_context), dim=-1)
        )
        if self.memory_type == "gru":
            context, next_hidden = self.memory(inputs, hidden_state)  # type: ignore[arg-type]
        else:
            context = self.memory(inputs)
            next_hidden = None
        candidates = self.candidate_projection(candidate_embeddings.float())
        interactions = torch.tanh(candidates + context[:, :, None, :])
        policy_logits = self.policy_scorer(interactions).squeeze(-1)
        q1 = self.q1_scorer(interactions).squeeze(-1)
        q2 = self.q2_scorer(interactions).squeeze(-1)
        masked_logits = policy_logits.masked_fill(~legal_mask.bool(), float("-inf"))
        return {
            "policy_logits": masked_logits,
            "action_log_probs": torch.log_softmax(masked_logits.float(), dim=-1),
            "q1": q1.masked_fill(~legal_mask.bool(), float("-inf")),
            "q2": q2.masked_fill(~legal_mask.bool(), float("-inf")),
            "values": self.value_scorer(context).squeeze(-1),
            "hidden_state": next_hidden,
        }

    def step(
        self,
        global_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        legal_mask: torch.Tensor,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        return self(
            global_embedding[:, None, :],
            candidate_embeddings[:, None, :, :],
            legal_mask[:, None, :],
            previous_action[:, None],
            previous_reward[:, None],
            hidden_state=hidden_state,
        )


def has_trajectory_head(checkpoint: str | Path | None) -> bool:
    return bool(checkpoint) and (Path(checkpoint) / TRAJECTORY_HEAD_FILENAME).is_file()


def create_trajectory_head(metadata: dict[str, Any]) -> TrajectoryPolicyHead:
    if metadata.get("trajectory_model_schema") != TRAJECTORY_MODEL_SCHEMA:
        raise ValueError("Trajectory checkpoint model schema does not match this code")
    return TrajectoryPolicyHead(
        int(metadata["trajectory_d_model"]),
        memory_type=str(metadata.get("trajectory_memory_type", "gru")),
        hidden_size=int(metadata.get("trajectory_hidden_size", 384)),
        recurrent_layers=int(metadata.get("trajectory_recurrent_layers", 2)),
        action_embedding_size=int(metadata.get("trajectory_action_embedding_size", 32)),
        reward_embedding_size=int(metadata.get("trajectory_reward_embedding_size", 16)),
        dropout=float(metadata.get("trajectory_dropout", 0.1)),
    )


def load_trajectory_head(
    checkpoint: str | Path,
    device: torch.device,
) -> TrajectoryPolicyHead:
    root = Path(checkpoint)
    metadata = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    head = create_trajectory_head(metadata).to(device)
    state = load_safetensors(root / TRAJECTORY_HEAD_FILENAME, device=str(device))
    head.load_state_dict(state)
    return head


def save_trajectory_head(head: TrajectoryPolicyHead, output_dir: str | Path) -> None:
    state = {
        key: value.detach().cpu().float().contiguous()
        for key, value in head.state_dict().items()
    }
    save_safetensors(state, Path(output_dir) / TRAJECTORY_HEAD_FILENAME)

