# Mechanics-conditioned policy (`mechanics-v1`)

> Legacy schema retained for checkpoint reproducibility. New runs default to
> [`mechanics-v2`](mechanics-v2.md), which fixes exact candidate collisions,
> restores compact identity signals, and adds more battle mechanics.

## Why this branch exists

The candidate-head run reached 36.52% validation action agreement at step 5,000
on its fixed 1,024-row sample, while training loss continued to move between
roughly 1.60 and 1.75. That was better than the older generative runs, but it was
also flat enough that finishing the remaining 44% of the epoch was hard to
justify.

The candidate head still asked a 0.5B language model to recover basic battle
mechanics from move names and prose. It had to learn that a name implied a type,
power, status effect, stat change, or switch cost before it could learn when the
action was useful. Those facts are deterministic and belong in the input.

`mechanics-v1` makes them explicit without expanding the prompt.

## Model shape

Each example has two branches:

```text
move-name-free battle state -> Qwen + LoRA -> one state vector

13 candidates x 97 numeric values -> mechanics MLP -> candidate vectors

state vector + each candidate vector -> shared scorer -> 13 logits -> legal mask
```

The mechanics tensor has shape `[batch, 13, 97]`. It is passed directly as a
floating-point tensor, so its 1,261 values add no tokenizer or attention tokens.
Rows for unavailable actions are zeroed and then removed from the probability
distribution by the same legal mask used during training and evaluation.

The state prompt keeps Pokémon species, HP, type, item, ability, base stats,
boosts, status, field state, team counts, and preview information. It removes
all move lines, prior move names, and legal candidate descriptions. On the first
100 prepared training rows, that reduced the median prompt from 1,211
`compact-v1` tokens to 466 `mechanics-v1` tokens.

## Candidate features

The versioned feature order is defined by `MECHANICS_FEATURE_NAMES` in
`src/pokemon_battler/mechanics.py`. The 97 values cover:

- action kind, forced-switch state, and Terastallization state;
- active HP and remaining-team fractions;
- move category, power, accuracy, priority, PP, expected hits, and STAB;
- type effectiveness buckets and common ability-based immunities;
- approximate damage range as a fraction of current target HP and approximate
  KO flags;
- fixed damage, healing, drain, recoil, self-destruction, pivoting, phazing,
  Protect behavior, critical-hit tendency, hazards, weather, and terrain;
- status, confusion, flinch, and expected self/target stat-stage changes;
- switch HP, entry-hazard damage, post-entry HP, status, base stats, relative
  speed, best known offensive pressure, and matchup against revealed opposing
  moves.

Move and Pokémon names may be used by the deterministic builder to retrieve
structured mechanics. Candidate move names are not handed to the model. If a
move cannot be resolved, the replay-provided basic values remain available and
`custom_effect_unmodeled` is set.

## Damage assumptions

The damage columns are pressure estimates, not exact Showdown calculations.
Prepared spectator rows omit EVs, IVs, natures, and some transient effects. The
builder therefore uses level, base stats, current stat stages, neutral nature,
31 IVs, and zero EVs. It includes base power, STAB, type effectiveness, expected
multi-hit count, current target HP, and the usual 0.85-to-1.00 random range. It
also applies burn's physical penalty and a conservative set of known ability
immunities.

This is enough to stop making Qwen infer “super effective” or approximate damage
from a move name. It is not enough to report exact live damage. The eventual
Showdown client should build the same schema from the live request and replace
estimated columns with simulator-backed values when the full state is known.

## Cache behavior

Feature construction is deterministic but too expensive to repeat inside every
training epoch. `pokemon-run` creates NumPy float16 memory maps beside each
JSONL file and reuses them when all of these match:

- mechanics schema;
- feature count and array shape;
- JSONL byte size, modification time, and row count.

The current split produces approximately 688 MiB of training features and
69 MiB of validation features. Cache files and partial files are gitignored.
Use `--rebuild-mechanics-cache` on `pokemon-run`, or `--overwrite` on
`pokemon-mechanics-cache`, after intentionally changing the source data without
changing its timestamp.

## Experiments this enables

The recommended hybrid run is:

```bash
.venv/bin/python -m pokemon_battler.experiment \
  --output-dir outputs/mechanics-v1-1epoch
```

It stops automatically after four validation checks without a 0.002 accuracy
gain. The best checkpoint still uses exact accuracy, with validation NLL as its
tiebreaker.

The mechanics-only ablation is:

```bash
pokemon-mechanics-baseline train \
  --train-file data/gen9ou-dev/train.jsonl \
  --validation-file data/gen9ou-dev/validation.jsonl \
  --output-dir outputs/mechanics-only-baseline
```

That network sees the same tensor and legal mask but no Qwen state embedding.
It answers a specific question: does the 0.5B model add useful strategic context
beyond the engineered mechanics? It is an ablation, not a decision to remove
the language model from the project.

## What a successful result means

The offline target is still the recorded human action. Accuracy is net top-1
agreement over every evaluated decision, not agreement after an oracle reveals
move versus switch. A better result would show that explicit mechanics make
behavior cloning easier. It would not yet demonstrate battle strength.

The next gate remains reproducible live battles against fixed opponents. Live
win rate, not replay imitation loss, decides whether the policy actually plays
better Pokémon.
