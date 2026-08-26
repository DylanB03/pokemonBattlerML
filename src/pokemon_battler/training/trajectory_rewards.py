from __future__ import annotations

from typing import Any


TERMINAL_REWARD = 1.0
DAMAGE_REWARD_SCALE = 0.02
FAINT_REWARD = 0.05
STATUS_REWARD = 0.02


def shaped_reward_from_event(event: dict[str, Any]) -> float:
    """Return a small, player-POV transition reward from a schema-3 history event.

    Winning remains the dominant signal.  The dense terms only make credit
    assignment less sparse; even a full health swing is much smaller than the
    terminal +/-1 reward.
    """
    player_delta = float(event.get("player_hp_delta", 0.0) or 0.0)
    opponent_delta = float(event.get("opponent_hp_delta", 0.0) or 0.0)
    reward = DAMAGE_REWARD_SCALE * (player_delta - opponent_delta)
    reward += FAINT_REWARD * (
        float(bool(event.get("opponent_fainted")))
        - float(bool(event.get("player_fainted")))
    )
    reward += STATUS_REWARD * (
        float(bool(event.get("opponent_status_inflicted")))
        - float(bool(event.get("player_status_inflicted")))
    )
    return reward


def terminal_reward(outcome: str | None) -> float:
    normalized = str(outcome or "").upper()
    if normalized == "WIN":
        return TERMINAL_REWARD
    if normalized == "LOSS":
        return -TERMINAL_REWARD
    return 0.0

