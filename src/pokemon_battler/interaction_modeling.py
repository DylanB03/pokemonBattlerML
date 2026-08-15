from __future__ import annotations

from typing import Any, Sequence

import torch

from pokemon_battler.actions import ACTION_COUNT
from pokemon_battler.interaction_features import (
    GLOBAL_ID_FIELDS,
    GLOBAL_NUMERIC_COUNT,
    HISTORY_ID_FIELDS,
    HISTORY_NUMERIC_NAMES,
    INTERACTION_MODEL_SCHEMA,
    INTERACTION_NAMESPACE_FIELDS,
    INTERACTION_VOCAB_SIZES,
    POKEMON_ID_FIELDS,
    POKEMON_NUMERIC_COUNT,
)
from pokemon_battler.mechanics_v2 import MECHANICS_FEATURE_COUNT, MECHANICS_IDENTITY_COUNT

ACTION_FAMILY_BY_ID = torch.tensor(
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    dtype=torch.long,
)
INTERACTION_HEAD_FILENAME = "interaction_head.safetensors"


class InteractionPolicyHead(torch.nn.Module):
    """Jointly encode the battle, rosters, history, and all legal candidates."""

    def __init__(
        self,
        qwen_hidden_size: int,
        *,
        d_model: int = 384,
        attention_heads: int = 8,
        layers: int = 4,
        feedforward_size: int = 1536,
        dropout: float = 0.1,
        identity_embedding_size: int = 16,
        qwen_mode: str = "lora",
    ) -> None:
        super().__init__()
        if qwen_mode not in {"lora", "frozen", "none"}:
            raise ValueError("qwen_mode must be lora, frozen, or none")
        if d_model % attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        self.schema = INTERACTION_MODEL_SCHEMA
        self.qwen_hidden_size = qwen_hidden_size
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.layers = layers
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.identity_embedding_size = identity_embedding_size
        self.qwen_mode = qwen_mode

        namespaces = {
            namespace
            for fields in INTERACTION_NAMESPACE_FIELDS.values()
            for namespace in fields
        }
        self.identity_embeddings = torch.nn.ModuleDict(
            {
                f"namespace_{namespace}": torch.nn.Embedding(
                    INTERACTION_VOCAB_SIZES[namespace],
                    identity_embedding_size,
                    padding_idx=0,
                )
                for namespace in sorted(namespaces)
            }
        )
        self.global_numeric = self._numeric_projector(GLOBAL_NUMERIC_COUNT)
        self.pokemon_numeric = self._numeric_projector(POKEMON_NUMERIC_COUNT)
        self.candidate_numeric = self._numeric_projector(MECHANICS_FEATURE_COUNT)
        self.history_numeric = self._numeric_projector(len(HISTORY_NUMERIC_NAMES))
        self.global_categorical = torch.nn.Linear(
            len(GLOBAL_ID_FIELDS) * identity_embedding_size, d_model
        )
        self.pokemon_categorical = torch.nn.Linear(
            len(POKEMON_ID_FIELDS) * identity_embedding_size, d_model
        )
        self.candidate_categorical = torch.nn.Linear(
            MECHANICS_IDENTITY_COUNT * identity_embedding_size, d_model
        )
        self.history_categorical = torch.nn.Linear(
            len(HISTORY_ID_FIELDS) * identity_embedding_size, d_model
        )
        self.qwen_norm = torch.nn.LayerNorm(qwen_hidden_size)
        self.qwen_projection = torch.nn.Linear(qwen_hidden_size, d_model)
        if qwen_mode == "none":
            self.qwen_norm.requires_grad_(False)
            self.qwen_projection.requires_grad_(False)
        # global, player active, player bench, opponent active, opponent bench,
        # history, normal move, switch, Tera move
        self.role_embedding = torch.nn.Embedding(9, d_model)
        self.actor_link = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.LayerNorm(d_model),
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=torch.nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.candidate_scorer = torch.nn.Linear(d_model, 1)
        # The policy scorer answers "what will I do?" while this separate head
        # answers "how likely is this legal action to lead to a win?".  Keeping
        # them separate is important: behavior-cloning logits are not action
        # values and cannot support an offline-RL or actor-critic objective.
        self.action_value_scorer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.GELU(),
            torch.nn.Linear(d_model // 2, 1),
        )
        self.family_scorer = torch.nn.Linear(d_model, 3)
        self.value_scorer = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.GELU(),
            torch.nn.Linear(d_model // 2, 1),
        )
        self._initialize_embeddings()

    def _numeric_projector(self, size: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(size, self.d_model),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.d_model),
        )

    def _initialize_embeddings(self) -> None:
        for embedding in self.identity_embeddings.values():
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()
        torch.nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)

    def config(self) -> dict[str, Any]:
        return {
            "interaction_model_schema": self.schema,
            "interaction_d_model": self.d_model,
            "interaction_attention_heads": self.attention_heads,
            "interaction_layers": self.layers,
            "interaction_feedforward_size": self.feedforward_size,
            "interaction_dropout": self.dropout,
            "interaction_identity_embedding_size": self.identity_embedding_size,
            "qwen_mode": self.qwen_mode,
            "interaction_vocab_sizes": dict(INTERACTION_VOCAB_SIZES),
        }

    def _embed_fields(
        self,
        ids: torch.Tensor,
        namespaces: Sequence[str],
    ) -> torch.Tensor:
        if ids.shape[-1] != len(namespaces):
            raise ValueError(
                f"Categorical tensor has {ids.shape[-1]} fields; expected {len(namespaces)}"
            )
        return torch.cat(
            [
                self.identity_embeddings[f"namespace_{namespace}"](ids[..., index])
                for index, namespace in enumerate(namespaces)
            ],
            dim=-1,
        )

    @staticmethod
    def _family_legal_mask(legal_mask: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                legal_mask[:, :4].any(dim=1),
                legal_mask[:, 4:9].any(dim=1),
                legal_mask[:, 9:13].any(dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def _hierarchical_log_probabilities(
        candidate_scores: torch.Tensor,
        family_logits: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        family_legal = InteractionPolicyHead._family_legal_mask(legal_mask)
        if not bool(family_legal.any(dim=1).all()):
            raise ValueError("Every interaction example must contain a legal action family")
        masked_family_logits = family_logits.masked_fill(~family_legal, float("-inf"))
        family_log_probs = torch.log_softmax(masked_family_logits.float(), dim=1)
        family_by_action = ACTION_FAMILY_BY_ID.to(candidate_scores.device)
        action_log_probs = torch.full_like(candidate_scores.float(), float("-inf"))
        for family_id in range(3):
            family_actions = family_by_action == family_id
            within_mask = legal_mask & family_actions[None, :]
            within_scores = candidate_scores.float().masked_fill(
                ~within_mask, float("-inf")
            )
            denominator = torch.logsumexp(within_scores, dim=1)
            denominator = torch.where(
                family_legal[:, family_id],
                denominator,
                torch.zeros_like(denominator),
            )
            values = (
                within_scores
                - denominator[:, None]
                + family_log_probs[:, family_id, None]
            )
            action_log_probs = torch.where(within_mask, values, action_log_probs)
        return action_log_probs, masked_family_logits

    def forward(
        self,
        state_hidden: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        global_numeric = batch["interaction_global_numeric"].float()
        global_ids = batch["interaction_global_ids"].long()
        pokemon_numeric = batch["interaction_pokemon_numeric"].float()
        pokemon_ids = batch["interaction_pokemon_ids"].long()
        pokemon_mask = batch["interaction_pokemon_mask"].bool()
        candidate_numeric = batch["interaction_candidate_numeric"].float()
        candidate_ids = batch["interaction_candidate_ids"].long()
        candidate_mask = batch["legal_action_mask"].bool()
        actor_slots = batch["interaction_candidate_actor_slot"].long()
        history_numeric = batch["interaction_history_numeric"].float()
        history_ids = batch["interaction_history_ids"].long()
        history_mask = batch["interaction_history_mask"].bool()
        batch_size = global_numeric.shape[0]

        global_token = self.global_numeric(global_numeric) + self.global_categorical(
            self._embed_fields(global_ids, INTERACTION_NAMESPACE_FIELDS["global"])
        )
        if self.qwen_mode != "none":
            global_token = global_token + self.qwen_projection(
                self.qwen_norm(state_hidden.float())
            )
        global_token = global_token + self.role_embedding.weight[0]

        pokemon_tokens = self.pokemon_numeric(pokemon_numeric) + self.pokemon_categorical(
            self._embed_fields(pokemon_ids, INTERACTION_NAMESPACE_FIELDS["pokemon"])
        )
        player_side = pokemon_numeric[..., 1] > 0.5
        active = pokemon_numeric[..., 2] > 0.5
        pokemon_roles = torch.where(
            player_side,
            torch.where(active, 1, 2),
            torch.where(active, 3, 4),
        )
        pokemon_tokens = pokemon_tokens + self.role_embedding(pokemon_roles)

        history_tokens = self.history_numeric(history_numeric) + self.history_categorical(
            self._embed_fields(history_ids, INTERACTION_NAMESPACE_FIELDS["history"])
        )
        history_tokens = history_tokens + self.role_embedding.weight[5]

        candidate_tokens = self.candidate_numeric(
            candidate_numeric
        ) + self.candidate_categorical(
            self._embed_fields(candidate_ids, INTERACTION_NAMESPACE_FIELDS["candidate"])
        )
        family_by_action = ACTION_FAMILY_BY_ID.to(candidate_tokens.device)
        candidate_roles = torch.where(
            family_by_action == 0,
            torch.tensor(6, device=candidate_tokens.device),
            torch.where(
                family_by_action == 1,
                torch.tensor(7, device=candidate_tokens.device),
                torch.tensor(8, device=candidate_tokens.device),
            ),
        )
        candidate_tokens = candidate_tokens + self.role_embedding(candidate_roles)[None, :, :]
        safe_slots = actor_slots.clamp(min=0, max=5)
        actor_tokens = torch.gather(
            pokemon_tokens[:, :6],
            1,
            safe_slots[..., None].expand(-1, -1, self.d_model),
        )
        actor_tokens = actor_tokens * (actor_slots >= 0)[..., None]
        candidate_tokens = candidate_tokens + self.actor_link(actor_tokens)

        tokens = torch.cat(
            (
                global_token[:, None, :],
                pokemon_tokens,
                history_tokens,
                candidate_tokens,
            ),
            dim=1,
        )
        padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
                ~pokemon_mask,
                ~history_mask,
                ~candidate_mask,
            ),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        global_output = encoded[:, 0]
        candidate_output = encoded[:, -ACTION_COUNT:]
        candidate_scores = self.candidate_scorer(candidate_output).squeeze(-1)
        family_logits = self.family_scorer(global_output)
        action_log_probs, masked_family_logits = self._hierarchical_log_probabilities(
            candidate_scores,
            family_logits,
            candidate_mask,
        )
        return {
            "action_log_probs": action_log_probs,
            "candidate_scores": candidate_scores.masked_fill(
                ~candidate_mask, float("-inf")
            ),
            "family_logits": masked_family_logits,
            "family_legal_mask": self._family_legal_mask(candidate_mask),
            "value_logits": self.value_scorer(global_output).squeeze(-1),
            "action_value_logits": self.action_value_scorer(candidate_output)
            .squeeze(-1)
            .masked_fill(~candidate_mask, float("-inf")),
            # These are the frozen per-turn representations consumed by the
            # trajectory policy.  Returning them here keeps the expensive Qwen
            # and interaction encoder identical between the old memoryless
            # policy, the cache builder, and live inference.
            "global_embedding": global_output,
            "candidate_embeddings": candidate_output,
        }


def interaction_policy_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    family_weights: torch.Tensor | None = None,
    family_aux_weight: float = 0.25,
    value_loss_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action_log_probs = outputs["action_log_probs"]
    action_targets = batch["action_ids"].to(action_log_probs.device)
    target_log_probs = action_log_probs.gather(1, action_targets[:, None])
    if not bool(torch.isfinite(target_log_probs).all()):
        raise ValueError("An interaction target is absent from its legal-action mask")
    policy_loss = torch.nn.functional.nll_loss(action_log_probs, action_targets)
    family_targets = batch["action_family_ids"].to(action_log_probs.device)
    family_loss = torch.nn.functional.cross_entropy(
        outputs["family_logits"],
        family_targets,
        weight=family_weights,
    )
    value_targets = batch["value_targets"].to(action_log_probs.device)
    value_mask = value_targets >= 0
    if value_loss_weight > 0 and bool(value_mask.any()):
        value_losses = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["value_logits"][value_mask].float(),
            value_targets[value_mask].float(),
            reduction="none",
        )
        weights = batch["value_weights"].to(action_log_probs.device)[value_mask].float()
        value_loss = (value_losses * weights).mean()
    else:
        value_loss = policy_loss.new_zeros(())
    total = policy_loss + family_aux_weight * family_loss + value_loss_weight * value_loss
    return total, {
        "policy_loss": policy_loss.detach(),
        "family_aux_loss": family_loss.detach(),
        "value_loss": value_loss.detach(),
    }
