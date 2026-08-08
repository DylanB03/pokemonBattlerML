# Proposed interaction policy v3

Status: design schema, not yet implemented.

This document fixes the implementation boundary for the next policy. It is
intended to prevent another long run in which the code, prepared data, cache,
and model silently disagree about which schema is being used.

The proposal keeps Qwen2.5-0.5B available, keeps every legal-action mask, and
reuses the mechanics-v2 candidate features. The fundamental change is that
Pokémon, recent events, and action candidates become tokens in a small
interaction transformer instead of being reduced to 13 independent MLP calls.

## What the source archive contains

The Metamon archive does not contain unrelated single-state examples. Each
compressed replay object contains two ordered arrays:

```text
states  = [state_0, state_1, ..., terminal_state]
actions = [action_0, action_1, ...]
```

The preparer pairs `states[:-1]` with `actions[:-1]`. Each state also contains
`player_prev_move` and `opponent_prev_move`. In a direct check of the first 100
raw trajectories, all 100 had ordered state/action arrays and all 4,051 states
contained both previous-move fields. Trajectory lengths ranged from 15 to 123
states in that sample.

Training still uses one decision row at a time. Preparation walks the complete
trajectory first, so it can attach information visible before that decision:

- recent moves from prior states;
- previously observed opponent moves, items, abilities, HP, and status;
- active-Pokémon changes inferred between adjacent states;
- HP, status, faint, and side-condition changes between adjacent states.

It must never attach a state or event that occurs after the target decision.
Turn-level sampling happens after the history trackers are updated, so a
selected 2% row may use preceding unsampled states without requiring those
states to become separate training rows.

The legacy `data/gen9ou-dev` JSONL contains isolated state snapshots plus the
immediate `*_prev_move` fields. It does not contain the accumulated history.
That information can be regenerated from the raw trajectory archive.

## Version boundaries

Three versions are recorded independently:

| Boundary | Proposed version | Meaning |
| --- | --- | --- |
| Prepared row | `schema_version: 3` | Decision-time state plus derived history events and stable rosters |
| Structured cache | `interaction-v1` | Fixed tensors described below |
| Model architecture | `interaction-policy-v1` | Token projections, interaction encoder, and output heads |

A model checkpoint must record all three versions. A cache or checkpoint load
must fail on a mismatch; it must not fall back to legacy fields.

## Prepared-row schema

Schema 3 extends the current schema-2 row rather than replacing its fields:

```json
{
  "schema_version": 3,
  "battle_id": "string",
  "battle_date": "YYYY-MM-DD",
  "rating": 1700,
  "outcome": "WIN",
  "source": "archive:member",
  "split": "train",
  "turn_index": 12,
  "state": {},
  "player_roster": [],
  "opponent_roster": [],
  "history_events": [],
  "action_id": 4,
  "target": "A4",
  "legal_action_ids": [0, 1, 2, 3, 4, 5],
  "legal_mask_quality": "recoverable"
}
```

### Stable rosters

`player_roster` and `opponent_roster` each contain up to six Pokémon. Slots are
sorted by normalized species name and remain stable for the trajectory.
Generation 9 OU species clause makes duplicate species an invalid normal case;
an occurrence is still resolved with a deterministic form-and-first-seen key.

The player roster is accumulated from the initial active Pokémon and available
switches, then retained after a Pokémon faints. The opponent roster begins with
team preview species and is filled with information as it is revealed. Unknown
item, ability, move, HP, Tera type, and status fields remain explicitly unknown;
they are never filled from Pokédex set guesses during preparation.

Every roster entry contains:

```json
{
  "slot": 0,
  "side": "player",
  "species": "greattusk",
  "present": true,
  "revealed": true,
  "active": false,
  "fainted": false,
  "hp_fraction": 0.73,
  "base_stats": {},
  "boosts": {},
  "types": [],
  "tera_type": "unknown",
  "terastallized": false,
  "status": "none",
  "item": "unknown",
  "ability": "unknown",
  "effects": [],
  "known_moves": []
}
```

### History events

Only the four most recent observable transitions are retained. A transition is
derived from `state[t - 1]` and the decision-time `state[t]`; it never uses
`state[t + 1]`.

```json
{
  "decision_offset": -1,
  "player_move": "earthquake",
  "opponent_move": "roost",
  "player_species_before": "greattusk",
  "player_species_after": "greattusk",
  "opponent_species_before": "corviknight",
  "opponent_species_after": "corviknight",
  "player_hp_delta": 0.0,
  "opponent_hp_delta": 0.42,
  "player_switched": false,
  "opponent_switched": false,
  "player_fainted": false,
  "opponent_fainted": false,
  "player_status_changed": false,
  "opponent_status_changed": false,
  "player_conditions_changed": false,
  "opponent_conditions_changed": false,
  "field_or_weather_changed": false
}
```

An unknown move remains `unknown`; a switch inferred from an active-species
change is represented by the switch fields, not invented as a move. Missing
decisions with action `-1` may contribute observable history but never become
policy targets.

## Structured cache

The cache is a directory rather than one monolithic matrix:

```text
train.interaction-v1/
  metadata.json
  global_numeric.f16.npy
  global_ids.u32.npy
  pokemon_numeric.f16.npy
  pokemon_ids.u32.npy
  pokemon_mask.u8.npy
  candidate_numeric.f16.npy
  candidate_ids.u32.npy
  candidate_mask.u8.npy
  candidate_actor_slot.i8.npy
  history_numeric.f16.npy
  history_ids.u32.npy
  history_mask.u8.npy
```

The first dimension of every array is the JSONL row count.

| Array | Shape | Purpose |
| --- | --- | --- |
| `global_numeric` | `[N, 30]` | Turn, remaining teams, side conditions, field, and legal-family availability |
| `global_ids` | `[N, 4]` | Format, weather, terrain, and legal-mask quality |
| `pokemon_numeric` | `[N, 12, 50]` | Public numeric state for six player and six opponent slots |
| `pokemon_ids` | `[N, 12, 11]` | Species, item, ability, types, Tera type, status, and four known moves |
| `pokemon_mask` | `[N, 12]` | Which roster slots exist |
| `candidate_numeric` | `[N, 13, 207]` | Existing mechanics-v2 action features |
| `candidate_ids` | `[N, 13, 32]` | Existing mechanics-v2 categorical identities |
| `candidate_mask` | `[N, 13]` | Legal actions only |
| `candidate_actor_slot` | `[N, 13]` | Player entity used by the move or entered by the switch |
| `history_numeric` | `[N, 4, 12]` | HP deltas and switch/faint/status/condition-change flags |
| `history_ids` | `[N, 4, 6]` | Two moves plus before/after active species on both sides |
| `history_mask` | `[N, 4]` | Which history positions exist |

### Global numeric fields

The 30 fields are:

```text
turn_fraction
forced_switch
can_tera
player_remaining_fraction
opponent_remaining_fraction
player_stealth_rock
player_spikes_layers
player_toxic_spikes_layers
player_sticky_web
player_reflect
player_light_screen
player_aurora_veil
player_tailwind
player_safeguard
opponent_stealth_rock
opponent_spikes_layers
opponent_toxic_spikes_layers
opponent_sticky_web
opponent_reflect
opponent_light_screen
opponent_aurora_veil
opponent_tailwind
opponent_safeguard
trick_room
player_team_hp_fraction
opponent_revealed_hp_fraction
opponent_roster_revealed_fraction
legal_move_fraction
legal_switch_fraction
legal_tera_fraction
```

### Pokémon numeric fields

The 50 fields are:

```text
present, player_side, active, revealed, fainted, hp_known,
item_known, ability_known, hp_fraction,
base_hp, base_atk, base_def, base_spa, base_spd, base_spe,
estimated_effective_speed, switch_entry_damage_fraction,
atk_stage, def_stage, spa_stage, spd_stage, spe_stage,
accuracy_stage, evasion_stage, known_moves_fraction,
terastallized, tera_type_known,
known_defensive_worst, known_move_immunity_fraction,
known_move_resistance_fraction, known_move_weakness_fraction,
opponent_moves_known_fraction, best_offensive_effectiveness,
best_damage_pressure,
effect_substitute, effect_protect, effect_leech_seed, effect_taunt,
effect_encore, effect_disable, effect_heal_block, effect_salt_cure,
effect_partial_trap, effect_yawn, effect_perish_song, effect_torment,
effect_confusion, effect_recharge, effect_ingrain, effect_magnet_rise
```

Categorical values use the same stable namespace vocabularies and zero-valued
unknown bucket as mechanics-v2. Numeric missingness is always paired with a
known/available flag; an unknown value must not look like a real zero.

The 12 history numeric fields are recency, player and opponent active-HP delta,
two switch flags, two faint flags, two status-change flags, two side-condition
change flags, and one field-or-weather-change flag. An active-HP delta is zero
when the corresponding active species changed; the switch flag distinguishes
that case from no damage.

### Cache metadata

`metadata.json` records:

```json
{
  "cache_schema": "interaction-v1",
  "prepared_schema_version": 3,
  "row_count": 286059,
  "source_path": "data/gen9ou-dev-schema3/train.jsonl",
  "source_size": 0,
  "source_mtime_ns": 0,
  "source_sha256": "sha256",
  "array_shapes": {},
  "numeric_feature_names": {},
  "categorical_field_names": {},
  "vocabulary_hashes": {},
  "generation": 9
}
```

The loader validates every shape, feature-name list, vocabulary hash, source
signature, and row count before exposing a memory map.

## Model input tokens

Each example contains at most 30 structured tokens:

```text
 1 global token
12 Pokémon tokens
 4 history-event tokens
13 candidate-action tokens
--
30 maximum
```

Unavailable roster, history, and candidate positions are padding-masked. An
illegal candidate cannot influence the policy score and receives negative
infinity before probability normalization.

### Token construction

Each token is projected to `d_model = 384`:

```text
numeric fields -> kind-specific MLP
categorical IDs -> namespace embeddings -> concatenation
token kind      -> learned kind embedding
side/role       -> learned role embedding
Qwen state      -> optional projected global contribution
```

No absolute roster-position embedding is used. Stable slot IDs exist to link a
switch candidate to its Pokémon, not to teach that alphabetic roster position
has strategic meaning.

For a move or Tera action, `candidate_actor_slot` points to the active player
Pokémon. For a switch, it points to the entering player roster slot. The actor
entity embedding is added to the initial candidate projection, giving the
network an explicit switch-to-Pokémon relationship before attention begins.

The compact Qwen prompt remains available only as a global residual input. It
does not contain generated damage prose. Three modes must be supported with the
same structured tensors:

```text
qwen_mode = lora
qwen_mode = frozen
qwen_mode = none
```

This is the controlled test of whether the 0.5B language model contributes
anything beyond the structured policy.

## Interaction encoder

The first implementation uses:

```text
d_model            384
attention heads       8
transformer layers    4
feed-forward width 1536
dropout             0.1
normalization      pre-norm
attention          full over valid structured tokens
```

Four layers add roughly 8-12 million trainable parameters including projections
and embeddings. The exact count must be printed at startup. This is small next
to Qwen2.5-0.5B and should fit the existing GPU budget with QLoRA.

The output candidate tokens produce 13 candidate scores. The output global
token produces three family logits and one optional state-value logit.

## Action hierarchy

The action families are fixed:

```text
ordinary move: A0-A3
switch:        A4-A8
Tera move:     A9-A12
```

A family is legal when at least one candidate in it is legal. The model forms a
proper hierarchical distribution:

```text
log P(action)
  = log P(family | state)
  + log P(action | family, state)
```

Family logits are masked to legal families. Candidate scores are normalized
only among legal actions in the selected family. Forced-switch rows therefore
have switch-family probability one, and unavailable Tera never participates.

The natural policy loss decomposes into:

```text
L_policy = CE_natural(family)
         + CE(within_target_family)
```

To prevent the roughly 2.4% Tera frequency from disappearing in the average,
training adds a small capped family-balanced auxiliary term:

```text
L = L_policy
  + 0.25 * CE_sqrt_inverse_capped_at_3(family)
  + 0.25 * BCE(value, battle_outcome)       # when value training is enabled
```

The exact-action term still follows the natural data distribution. The
balanced term is deliberately auxiliary; it must not redefine Tera as one
third of normal play. Family precision, recall, calibration, and overall action
accuracy decide whether its coefficient is retained.

## Value head

The global output token passes through a two-layer MLP to one win logit. `WIN`
is target one and `LOSS` target zero from that row's player perspective.

Rows are weighted so that a long battle does not contribute more total value
loss merely because it has more decisions. The initial value model is assessed
with log loss, Brier score, ROC AUC, and calibration against a constant win-rate
baseline. Its output is not called win probability until calibration has been
measured.

The value head is auxiliary in the first policy run. Offline RL and search are
separate later stages; they are not silently enabled by this schema.

## Training stages

Each stage changes one major variable:

1. Prepare schema-3 rows and build the interaction cache.
2. Verify strict no-future-leakage and action-to-roster mappings with unit tests.
3. Overfit 128 rows to at least 95% exact action accuracy.
4. Train the interaction policy with natural action loss only.
5. Add the hierarchical family loss and compare on the identical validation
   sample.
6. Add the value loss only after policy behavior is understood.
7. Compare `qwen_mode=lora`, `frozen`, and `none` with identical structured
   tensors and budgets.

No test split is opened during these stages.

## Required reports

Every validation report includes:

- exact, top-2, and top-3 action agreement;
- candidate NLL and mean reciprocal rank;
- ordinary-move, switch, and Tera precision and recall;
- family confusion matrix;
- results split by exact, PP-aware, and recoverable legal masks;
- early-, middle-, and late-battle slices;
- performance by number of revealed opponent moves and Pokémon;
- value metrics and reliability bins when the value head is enabled;
- Qwen mode, trainable parameter count, peak VRAM, examples per second, and
  elapsed time.

## Acceptance rules

The architecture proceeds only if:

- the 128-row gate reaches at least 95%;
- illegal-action probability is exactly zero;
- permuting roster storage order while preserving slot relationships does not
  change output beyond numerical tolerance;
- masking a padding token does not change valid action scores;
- the history builder demonstrably uses no future state;
- interaction-policy validation exceeds the mechanics-v2 fixed-sample result,
  or supplies a clear family/value improvement without a material top-1 loss;
- Tera recall rises without an unacceptable false-Tera rate.

The first live-policy candidate is chosen only after a fixed-opponent battle
evaluation. Replay action agreement remains a diagnostic rather than a claim of
battle strength.
