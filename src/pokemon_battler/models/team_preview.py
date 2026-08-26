from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from pokemon_battler.models.interaction_features import INTERACTION_VOCAB_SIZES, identity

TEAM_PREVIEW_SCHEMA = "foul-play-team-preview-v1"
TEAM_PREVIEW_HEAD_FILENAME = "team_preview_head.safetensors"
PREVIEW_SLOTS = 6
PREVIEW_NUMERIC_COUNT = 7
PREVIEW_ID_FIELDS = ("species", "type", "type", "item", "ability")


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _types(pokemon: dict[str, Any]) -> list[str]:
    value = pokemon.get("types") or []
    values = value.split() if isinstance(value, str) else list(value)
    return [str(item) for item in values[:2]] + ["notype"] * max(0, 2 - len(values))


def _pokemon_features(pokemon: dict[str, Any], *, opponent: bool) -> tuple[list[float], list[int]]:
    types = _types(pokemon)
    numeric = [
        float(pokemon.get(f"base_{stat}", 0) or 0) / 255.0
        for stat in ("hp", "atk", "def", "spa", "spd", "spe")
    ] + [float(opponent)]
    ids = [
        identity("species", pokemon.get("base_species") or pokemon.get("name")),
        identity("type", types[0]),
        identity("type", types[1]),
        identity("item", pokemon.get("item") if not opponent else "unknownitem"),
        identity("ability", pokemon.get("ability") if not opponent else "unknownability"),
    ]
    return numeric, ids


def preview_features(row: dict[str, Any]) -> dict[str, Any]:
    player = sorted(row["player_roster"], key=lambda pokemon: int(pokemon.get("slot", 99)))
    opponent = sorted(
        row["opponent_roster"], key=lambda pokemon: int(pokemon.get("slot", 99))
    )
    if not 1 <= len(player) <= PREVIEW_SLOTS or not 1 <= len(opponent) <= PREVIEW_SLOTS:
        raise ValueError("Team preview needs between one and six Pokémon per side")
    numeric: list[list[float]] = []
    ids: list[list[int]] = []
    mask: list[bool] = []
    for side, members in ((False, player), (True, opponent)):
        for slot in range(PREVIEW_SLOTS):
            if slot < len(members):
                values, categories = _pokemon_features(members[slot], opponent=side)
                numeric.append(values)
                ids.append(categories)
                mask.append(True)
            else:
                numeric.append([0.0] * PREVIEW_NUMERIC_COUNT)
                ids.append([0] * len(PREVIEW_ID_FIELDS))
                mask.append(False)
    return {"numeric": numeric, "ids": ids, "mask": mask, "player_count": len(player)}


def render_preview_prompt(row: dict[str, Any]) -> str:
    def member(pokemon: dict[str, Any], *, own: bool) -> str:
        pieces = [
            str(pokemon.get("base_species") or pokemon.get("name") or "unknownpokemon"),
            "/".join(_types(pokemon)),
            "stats=" + "/".join(
                str(int(pokemon.get(f"base_{stat}", 0) or 0))
                for stat in ("hp", "atk", "def", "spa", "spd", "spe")
            ),
        ]
        if own:
            pieces.extend(
                (
                    f"item={pokemon.get('item', 'unknownitem')}",
                    f"ability={pokemon.get('ability', 'unknownability')}",
                    "moves=" + "/".join(
                        str(move.get("name") or "nomove")
                        for move in pokemon.get("moves", [])
                    ),
                )
            )
        return "|".join(pieces)

    player = sorted(row["player_roster"], key=lambda pokemon: int(pokemon.get("slot", 99)))
    opponent = sorted(
        row["opponent_roster"], key=lambda pokemon: int(pokemon.get("slot", 99))
    )
    return (
        "Choose the lead that maximizes the probability of winning this Pokemon battle.\n"
        + "OWN_TEAM\n"
        + "\n".join(member(pokemon, own=True) for pokemon in player)
        + "\nOPPONENT_PREVIEW\n"
        + "\n".join(member(pokemon, own=False) for pokemon in opponent)
    )


class TeamPreviewCollator:
    def __init__(self, tokenizer: Any, *, max_length: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [
            self.tokenizer.encode(render_preview_prompt(row), add_special_tokens=True)
            for row in rows
        ]
        if any(len(values) > self.max_length for values in encoded):
            raise ValueError("Team-preview prompt exceeds max_length")
        length = max(map(len, encoded))
        features = [preview_features(row) for row in rows]
        return {
            "input_ids": torch.tensor(
                [values + [self.pad_token_id] * (length - len(values)) for values in encoded]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(values) + [0] * (length - len(values)) for values in encoded]
            ),
            "preview_numeric": torch.tensor([item["numeric"] for item in features]),
            "preview_ids": torch.tensor([item["ids"] for item in features]),
            "preview_mask": torch.tensor([item["mask"] for item in features]),
            "preview_player_count": torch.tensor([item["player_count"] for item in features]),
            "preview_targets": torch.tensor([int(row.get("action_id", -1)) for row in rows]),
            "preview_teacher_policy": torch.tensor(
                [
                    list((row.get("teacher") or {}).get("policy") or [])
                    + [0.0] * (PREVIEW_SLOTS - len((row.get("teacher") or {}).get("policy") or []))
                    for row in rows
                ],
                dtype=torch.float32,
            ),
        }


class TeamPreviewHead(torch.nn.Module):
    def __init__(self, qwen_hidden_size: int, *, d_model: int = 256) -> None:
        super().__init__()
        self.qwen_hidden_size = qwen_hidden_size
        self.d_model = d_model
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(INTERACTION_VOCAB_SIZES[namespace], 16, padding_idx=0)
                for namespace in PREVIEW_ID_FIELDS
            ]
        )
        self.numeric = torch.nn.Linear(PREVIEW_NUMERIC_COUNT, d_model)
        self.categorical = torch.nn.Linear(len(PREVIEW_ID_FIELDS) * 16, d_model)
        self.qwen = torch.nn.Linear(qwen_hidden_size, d_model)
        self.side = torch.nn.Embedding(2, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, 3, norm=torch.nn.LayerNorm(d_model))
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, d_model), torch.nn.GELU(), torch.nn.Linear(d_model, 1)
        )

    def forward(self, qwen_hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        numeric = batch["preview_numeric"].float()
        ids = batch["preview_ids"].long()
        mask = batch["preview_mask"].bool()
        categorical = torch.cat(
            [embedding(ids[..., index]) for index, embedding in enumerate(self.embeddings)],
            dim=-1,
        )
        side_ids = torch.cat(
            (
                torch.zeros(PREVIEW_SLOTS, dtype=torch.long, device=ids.device),
                torch.ones(PREVIEW_SLOTS, dtype=torch.long, device=ids.device),
            )
        )
        tokens = self.numeric(numeric) + self.categorical(categorical) + self.side(side_ids)[None]
        tokens = tokens + self.qwen(qwen_hidden.float())[:, None, :]
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        own = encoded[:, :PREVIEW_SLOTS]
        pooled = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        logits = self.scorer(
            torch.cat((own, pooled[:, None].expand(-1, PREVIEW_SLOTS, -1)), dim=-1)
        ).squeeze(-1)
        own_mask = mask[:, :PREVIEW_SLOTS]
        return logits.masked_fill(~own_mask, float("-inf"))


def save_team_preview_head(head: TeamPreviewHead, output_dir: str | Path) -> None:
    state = {
        key: value.detach().cpu().float().contiguous() for key, value in head.state_dict().items()
    }
    save_safetensors(state, Path(output_dir) / TEAM_PREVIEW_HEAD_FILENAME)


def load_team_preview_head(
    qwen_hidden_size: int, checkpoint: str | Path, device: torch.device
) -> TeamPreviewHead:
    path = Path(checkpoint) / TEAM_PREVIEW_HEAD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata_path = Path(checkpoint) / "training_config.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    head = TeamPreviewHead(
        qwen_hidden_size, d_model=int(metadata.get("team_preview_d_model", 256))
    ).to(device=device, dtype=torch.float32)
    head.load_state_dict(load_safetensors(path, device=str(device)))
    return head


def has_team_preview_head(checkpoint: str | Path) -> bool:
    return (Path(checkpoint) / TEAM_PREVIEW_HEAD_FILENAME).is_file()


def live_preview_observation(battle: Any) -> dict[str, Any]:
    """Convert poke-env's preview request to the teacher's public schema."""
    from pokemon_battler.showdown.live_state import _pokemon_to_state

    player_members = list(getattr(battle, "teampreview_team", None) or battle.team.values())
    opponent_members = list(getattr(battle, "teampreview_opponent_team", None) or [])

    def roster(members: list[Any], side_name: str) -> list[dict[str, Any]]:
        rows = []
        for slot, pokemon in enumerate(members):
            row = _pokemon_to_state(pokemon)
            row.update(
                {
                    "slot": slot,
                    "side": side_name,
                    "active": False,
                    "present": True,
                    "revealed": side_name == "player",
                    "fainted": False,
                }
            )
            if side_name == "opponent":
                row.update(
                    {
                        "hp_pct": None,
                        "item": "unknownitem",
                        "ability": "unknownability",
                        "tera_type": "notype",
                        "moves": [],
                    }
                )
            rows.append(row)
        return rows

    player_roster = roster(player_members, "player")
    opponent_roster = roster(opponent_members, "opponent")
    return {
        "preview_schema": TEAM_PREVIEW_SCHEMA,
        "decision_phase": "team_preview",
        "state": {
            "format": str(getattr(battle, "format", None) or "gen9ou"),
            "opponent_teampreview": [row["name"] for row in opponent_roster],
        },
        "player_roster": player_roster,
        "opponent_roster": opponent_roster,
        "legal_action_ids": list(range(len(player_roster))),
    }
