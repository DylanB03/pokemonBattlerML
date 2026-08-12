from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from pokemon_battler.actions import action_label, describe_action
from pokemon_battler.live_state import (
    LiveBattleTracker,
    live_action_to_order,
)
from pokemon_battler.modeling import (
    has_interaction_head,
    indexed_logits_parameter,
    interaction_outputs,
    load_interaction_head,
    load_policy_model,
    load_training_metadata,
)
from pokemon_battler.training_data import InteractionInferenceCollator


_HTML_TAG = re.compile(r"<[^>]*>")
_RATING_UPDATE = re.compile(
    r"^(?P<username>.+?)'s rating:\s*(?P<before>\d+)\s*"
    r"(?:→|->)\s*(?P<after>\d+)\s*"
    r"\((?P<change>[+-]\d+)\s+for\s+"
    r"(?P<result>winning|losing|tying)\)\s*$"
)


def _showdown_id(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def parse_showdown_rating_update(
    raw_message: str,
    *,
    username: str,
) -> dict[str, Any] | None:
    """Parse this player's exact old-to-new ELO transition from a ladder result."""
    plain_text = _HTML_TAG.sub("", unescape(raw_message)).strip()
    match = _RATING_UPDATE.fullmatch(plain_text)
    if match is None or _showdown_id(match.group("username")) != _showdown_id(username):
        return None
    before = int(match.group("before"))
    after = int(match.group("after"))
    # Derive the delta from the two authoritative displayed ratings. This also
    # protects the summary if Showdown ever changes the explanatory text.
    return {
        "before": before,
        "after": after,
        "change": after - before,
        "result": match.group("result"),
    }


@dataclass(frozen=True)
class InteractionPrediction:
    action_id: int
    log_probabilities: dict[int, float]
    preferences: dict[int, float]
    entropy: float
    value_probability: float
    latency_seconds: float


class InteractionPolicyRuntime:
    """Load one frozen checkpoint and score unlabelled live decisions."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_name: str | None = None,
        max_length: int | None = None,
        prompt_format: str | None = None,
        dtype: str | None = None,
        load_in_4bit: bool | None = None,
        local_files_only: bool | None = None,
        attn_implementation: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not has_interaction_head(self.checkpoint):
            raise FileNotFoundError(
                f"Interaction checkpoint is missing interaction_head.safetensors: "
                f"{self.checkpoint}"
            )
        metadata = load_training_metadata(self.checkpoint)
        self.metadata = metadata
        self.model_name = model_name or str(metadata.get("model", "Qwen/Qwen2.5-0.5B"))
        self.max_length = max_length or int(metadata.get("max_length", 4096))
        self.prompt_format = prompt_format or str(
            metadata.get("prompt_format", "mechanics-v2")
        )
        resolved_dtype = dtype or str(metadata.get("dtype", "auto"))
        resolved_4bit = (
            bool(metadata.get("load_in_4bit", False))
            if load_in_4bit is None
            else load_in_4bit
        )
        resolved_local = (
            bool(metadata.get("local_files_only", False))
            if local_files_only is None
            else local_files_only
        )
        resolved_attention = attn_implementation or str(
            metadata.get("attn_implementation", "auto")
        )
        adapter_path = (
            str(self.checkpoint)
            if (self.checkpoint / "adapter_config.json").is_file()
            else None
        )
        self.model, self.tokenizer, self.device = load_policy_model(
            self.model_name,
            adapter_path=adapter_path,
            dtype=resolved_dtype,
            load_in_4bit=resolved_4bit,
            local_files_only=resolved_local,
            attn_implementation=resolved_attention,
        )
        self.model.eval()
        self.head = load_interaction_head(self.model, self.checkpoint, self.device)
        self.head.eval()
        self.collator = InteractionInferenceCollator(
            self.tokenizer,
            max_length=self.max_length,
            truncation="error",
            prompt_format=self.prompt_format,
        )
        self.logits_parameter = indexed_logits_parameter(self.model)

    def predict(
        self,
        row: dict[str, Any],
        *,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> InteractionPrediction:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        started = perf_counter()
        batch = {
            key: value.to(self.device)
            for key, value in self.collator([row]).items()
        }
        with torch.inference_mode():
            outputs = interaction_outputs(
                self.model,
                self.head,
                batch,
                logits_parameter=self.logits_parameter,
            )
        action_log_probs = outputs["action_log_probs"][0].float().cpu()
        legal = [int(value) for value in row["legal_action_ids"]]
        log_probabilities = {
            action_id: float(action_log_probs[action_id].item()) for action_id in legal
        }
        if not all(math.isfinite(value) for value in log_probabilities.values()):
            raise ValueError("Interaction policy returned a non-finite legal-action score")
        sampling_log_probabilities = {
            action_id: log_probability / temperature
            for action_id, log_probability in log_probabilities.items()
        }
        maximum = max(sampling_log_probabilities.values())
        preferences = {
            action_id: math.exp(log_probability - maximum)
            for action_id, log_probability in sampling_log_probabilities.items()
        }
        total = sum(preferences.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Interaction policy returned an invalid preference distribution")
        preferences = {
            action_id: probability / total
            for action_id, probability in preferences.items()
        }
        # PPO must retain the probability distribution that actually sampled
        # the action, including any rollout temperature.
        log_probabilities = {
            action_id: math.log(probability)
            for action_id, probability in preferences.items()
        }
        if sample:
            legal_ids = list(preferences)
            sampled_index = int(
                torch.multinomial(
                    torch.tensor(
                        [preferences[action_id] for action_id in legal_ids],
                        dtype=torch.float32,
                    ),
                    1,
                ).item()
            )
            action_id = legal_ids[sampled_index]
        else:
            action_id = max(preferences, key=preferences.get)
        entropy = -sum(
            probability * math.log(probability)
            for probability in preferences.values()
            if probability > 0
        )
        value_probability = float(
            torch.sigmoid(outputs["value_logits"][0].float()).cpu().item()
        )
        return InteractionPrediction(
            action_id=action_id,
            log_probabilities=log_probabilities,
            preferences=preferences,
            entropy=entropy,
            value_probability=value_probability,
            latency_seconds=perf_counter() - started,
        )


class DecisionTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")


def deterministic_teampreview(battle: AbstractBattle) -> str:
    """Keep the submitted team order and lead with slot one for reproducible tests."""
    members = list(range(1, len(battle.team) + 1))
    for pokemon in battle.team.values():
        pokemon._selected_in_teampreview = True
    return "/team " + "".join(str(member) for member in members)


class DeterministicPreviewMixin:
    def teampreview(self, battle: AbstractBattle) -> str:
        return deterministic_teampreview(battle)


class InteractionPlayer(Player):
    def __init__(
        self,
        runtime: InteractionPolicyRuntime,
        *,
        trace_writer: DecisionTraceWriter | None = None,
        fail_fast: bool = False,
        sample_actions: bool = False,
        sampling_temperature: float = 1.0,
        team_preview_policy: str = "first",
        decision_callback: Callable[[dict[str, Any]], None] | None = None,
        **player_kwargs: Any,
    ) -> None:
        self.runtime = runtime
        self.trace_writer = trace_writer
        self.fail_fast = fail_fast
        self.sample_actions = sample_actions
        self.sampling_temperature = sampling_temperature
        if team_preview_policy not in {"first", "random"}:
            raise ValueError("team_preview_policy must be 'first' or 'random'")
        self.team_preview_policy = team_preview_policy
        self.decision_callback = decision_callback
        self.trackers: dict[str, LiveBattleTracker] = {}
        self.rating_updates: dict[str, dict[str, Any]] = {}
        self.decision_count = 0
        self.fallback_count = 0
        self.inference_latencies: list[float] = []
        super().__init__(**player_kwargs)

    def teampreview(self, battle: AbstractBattle) -> str:
        if self.team_preview_policy == "random":
            return self.random_teampreview(battle)
        return deterministic_teampreview(battle)

    def _capture_rating_updates(self, split_messages: list[list[str]]) -> None:
        if not split_messages or not split_messages[0]:
            return
        battle_id = split_messages[0][0].removeprefix(">")
        for split_message in split_messages[1:]:
            if len(split_message) < 3 or split_message[1] != "raw":
                continue
            update = parse_showdown_rating_update(
                "|".join(split_message[2:]),
                username=self.username,
            )
            if update is not None:
                self.rating_updates[battle_id] = update

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        # poke-env marks the game complete as soon as it reads |win|/|tie|, but
        # Showdown's exact old -> new rating line follows later in the same
        # message. Capture the whole message first so closing a finished batch
        # cannot race and lose the final game's ELO update.
        self._capture_rating_updates(split_messages)
        await super()._handle_battle_message(split_messages)

    def _write_decision(
        self,
        *,
        battle: AbstractBattle,
        row: dict[str, Any] | None,
        prediction: InteractionPrediction | None,
        order: BattleOrder,
        fallback_reason: str | None,
    ) -> None:
        if self.trace_writer is None:
            return
        record: dict[str, Any] = {
            "event": "decision",
            "battle_id": battle.battle_tag,
            "showdown_turn": int(getattr(battle, "turn", 0) or 0),
            "request_id": (getattr(battle, "last_request", None) or {}).get("rqid"),
            "order": str(order),
            "fallback_reason": fallback_reason,
        }
        if row is not None:
            record["observation"] = row
        if prediction is not None:
            record["prediction"] = {
                "action_id": prediction.action_id,
                "action_label": action_label(prediction.action_id),
                "action": describe_action(row["state"], prediction.action_id),
                "log_probabilities": {
                    action_label(key): value
                    for key, value in prediction.log_probabilities.items()
                },
                "preferences": {
                    action_label(key): value
                    for key, value in prediction.preferences.items()
                },
                "entropy": prediction.entropy,
                "value_probability": prediction.value_probability,
                "latency_seconds": prediction.latency_seconds,
            }
        self.trace_writer.write(record)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        tracker = self.trackers.setdefault(
            battle.battle_tag,
            LiveBattleTracker(battle_id=battle.battle_tag),
        )
        row: dict[str, Any] | None = None
        prediction: InteractionPrediction | None = None
        fallback_reason: str | None = None
        try:
            row = tracker.observe(battle)
            prediction = self.runtime.predict(
                row,
                sample=self.sample_actions,
                temperature=self.sampling_temperature,
            )
            order = live_action_to_order(battle, prediction.action_id)
            if order is None:
                raise ValueError(
                    f"Predicted {action_label(prediction.action_id)} could not be mapped "
                    "back to the current Showdown request"
                )
            self.inference_latencies.append(prediction.latency_seconds)
            if self.decision_callback is not None:
                self.decision_callback(
                    {
                        "event": "decision",
                        "battle_id": battle.battle_tag,
                        "observation": row,
                        "action_id": prediction.action_id,
                        "old_log_probability": prediction.log_probabilities[
                            prediction.action_id
                        ],
                        "value_probability": prediction.value_probability,
                    }
                )
        except Exception as exc:
            if self.fail_fast:
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
            self.logger.exception("Live policy fallback in %s", battle.battle_tag)
            order = self.choose_random_move(battle)
            self.fallback_count += 1
        self.decision_count += 1
        self._write_decision(
            battle=battle,
            row=row,
            prediction=prediction,
            order=order,
            fallback_reason=fallback_reason,
        )
        return order

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        rating_update = getattr(self, "rating_updates", {}).get(battle.battle_tag)
        if self.decision_callback is not None:
            self.decision_callback(
                {
                    "event": "battle_finished",
                    "battle_id": battle.battle_tag,
                    "won": battle.won is True,
                    "lost": battle.lost is True,
                    "opponent": battle.opponent_username,
                    "turns": battle.turn,
                    "rating": battle.rating,
                    "opponent_rating": battle.opponent_rating,
                    "rating_update": rating_update,
                }
            )
        if self.trace_writer is not None:
            self.trace_writer.write(
                {
                    "event": "battle_finished",
                    "battle_id": battle.battle_tag,
                    "won": battle.won,
                    "lost": battle.lost,
                    "opponent": battle.opponent_username,
                    "turns": battle.turn,
                    "rating": battle.rating,
                    "opponent_rating": battle.opponent_rating,
                    "rating_update": rating_update,
                }
            )
