# Mechanics-conditioned policy (`mechanics-v2`)

## Why v1 was replaced

`mechanics-v1` proved that deterministic battle facts can bypass the tokenizer,
but its 97 numeric columns compressed too many distinct actions into the same
representation. A full scan of the 286,059-row training cache found 2,880 rows
with at least two legal candidates whose v1 vectors were exactly identical.
The recorded target belonged to one of those collision groups in 780 rows.
Examples included Rest versus Sleep Talk, Reflect versus Light Screen, Spikes
versus Stealth Rock, and moves with distinct custom behavior but similar power
and type summaries.

That is a representation error. Cross-entropy cannot learn to prefer one
candidate when the scorer receives the same state vector and the same candidate
vector for both. A longer run or a different learning rate cannot recover
information that is absent from the input.

`mechanics-v2` keeps the useful numeric branch, adds explicit battle context,
and gives exact categorical identities their own learned embeddings. It is
designed for the existing Qwen2.5-0.5B model; it does not require a model
upgrade, retrieval, active learning, continual learning, or simulator RL.

## Model input

Each example has three inputs:

```text
compact battle state -> Qwen + LoRA -> one state vector

13 candidates x 207 numeric values -> numeric mechanics encoder

13 candidates x 32 categorical IDs -> shared identity embeddings

state + numeric mechanics + identities -> shared scorer -> legal mask
```

The numeric cache is a float16 tensor with shape `[rows, 13, 207]`. Categorical
IDs are rebuilt from the JSONL state by the collator. They are not interpreted
as ordered numbers. Each namespace has a trainable embedding table, and fields
that share a namespace also share the table. For example, the candidate move,
last player move, last opponent move, and known move slots all use the same move
embedding space.

Unavailable A0-A12 rows are zero-filled. The legal-action mask is then applied
to logits before both cross-entropy and argmax, so illegal actions receive no
probability and cannot be selected.

## Numeric mechanics

V2 retains all 97 v1 columns and adds 110 columns. The additions include:

- active and opposing base stats and all seven current stat stages;
- separate Stealth Rock, Spikes layers, Toxic Spikes layers, Sticky Web,
  Reflect, Light Screen, Aurora Veil, Tailwind, and Safeguard fields for each
  side;
- Trick Room, estimated effective speed, and estimated move order;
- contact, sound, bullet, pulse, punch, bite, dance, slicing, wind, powder,
  reflectable, and Substitute-bypass move flags;
- separate hazard-setting, screen-setting, Tailwind, Safeguard, hazard-control,
  and side-condition-swap behaviors;
- conditional priority, item interaction, called moves, delayed attacks and
  healing, full recovery, status curing, sleep requirements, Substitute, field
  removal, and locked-move behavior;
- distinct common volatile effects such as Leech Seed, Taunt, Encore, Disable,
  Heal Block, Salt Cure, trapping, Yawn, Perish Song, and Torment;
- known opposing-move defensive pressure for ordinary moves, switches, and the
  candidate Terastallized type;
- corrected damage paths for Psyshock/Psystrike/Secret Sword, Body Press, Foul
  Play, common variable-power moves, weather, terrain, burn, common offensive
  and defensive items, and common abilities;
- original-type STAB retained after Terastallization, rather than incorrectly
  treating every non-Tera-type attack as losing STAB.

Damage remains an estimate. Replays omit EVs, IVs, natures, exact sets, and some
temporary state. The features are intended to encode pressure and relative
candidate value, not reproduce the Showdown damage calculator exactly.

## Categorical identities

The 32 fields cover:

- candidate move;
- actor and opponent species, item, ability, types, status, and active effect;
- actor Tera type and candidate move type;
- weather, terrain, move target, side condition, and volatile effect;
- previous player and opponent move;
- four known actor moves and four known opponent moves.

Moves and species use stable vocabularies derived from the Generation 9
`poke-env` data. Weather, terrain, effects, side conditions, and move targets
use their exact simulator enums, with reserved fallback buckets for values added
by later data. Abilities are collected from the Pokédex with fallback buckets
for unusual values. Items use a large deterministic namespace-specific hash
space because this `poke-env` version does not expose a complete item catalog.
Zero is reserved for missing or unknown values. These identities let the model
retain exceptions without pretending that every mechanic can be represented by
one `has effect` flag.

## Prompt policy

V1 removed every move name from Qwen's prompt. V2 restores compact move-name
lists, prior moves, and short history while continuing to omit verbose move
attributes and candidate descriptions. Pokémon names remain present alongside
their stats, types, item, ability, status, effects, and boosts.

On the first 100 prepared training rows, v2 had a 592-token median and 707-token
maximum. The equivalent `compact-v1` prompts measured 1,207 and 1,415 tokens.
V2 is about 129 median tokens longer than the name-free v1 prompt, not several
times longer than it.

This is intentional redundancy. Numeric features encourage generalization by
mechanics; exact identities and compact names preserve residual information for
special moves and interactions. The model does not need to infer damage or type
effectiveness from names alone.

## Compatibility and checkpoint isolation

The schema name is part of cache and checkpoint metadata. V1 and v2 differ in
feature width, head shape, prompt format, and identity inputs. The loader reads
that metadata and reconstructs the corresponding head:

- old `mechanics-v1` caches and checkpoints remain loadable;
- a v2 run creates `*.mechanics-v2.npy` caches and does not modify v1 caches;
- a new output directory is required unless `--overwrite-output` is explicitly
  passed;
- `best/` and `final/` remain separate complete post-trained checkpoints.

Do not point a v2 run at a v1 cache. The loader rejects mismatched schemas
instead of silently reading the wrong columns.

## Run it

The single-command experiment is:

```bash
.venv/bin/python -m pokemon_battler.experiment \
  --output-dir outputs/mechanics-v2-1epoch
```

On this repository's current split, the first numeric-cache build needs about
1.43 GiB for training and 147 MiB for validation. Subsequent runs reuse a cache
when the schema, source signature, row count, feature count, and array shape all
match.

The training loss is the same legal-candidate cross-entropy used by the
candidate-head and mechanics-v1 runs. Its scale is therefore comparable, but
early training values still fluctuate with batch composition. Select the model
by fixed validation accuracy, using candidate NLL as the tiebreaker, rather than
by whichever individual training minibatch happens to have the lowest loss.

The mechanics-only ablation uses the same 207 numeric values, 32 identity
fields, and legal mask without Qwen:

```bash
pokemon-mechanics-baseline train \
  --train-file data/gen9ou-dev/train.jsonl \
  --validation-file data/gen9ou-dev/validation.jsonl \
  --output-dir outputs/mechanics-v2-only-baseline
```

That comparison tests whether Qwen contributes useful strategic state context.
It does not change the recommended hybrid architecture.
