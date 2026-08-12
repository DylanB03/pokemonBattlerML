# Local Showdown evaluation

The local evaluation path runs the existing interaction checkpoint through
complete battles on the official Pokémon Showdown simulator. It does not train,
finetune, or overwrite the checkpoint.

## One command

From the repository root in the active Python environment:

```bash
python -m pokemon_battler.live_eval --games 20 --opponent heuristic
```

The equivalent installed entry point is:

```bash
pokemon-live-eval --games 20 --opponent heuristic
```

For a fast installation and integration check before a longer comparison, run:

```bash
python -m pokemon_battler.live_eval --games 1 --opponent random --fail-fast
```

The first run bootstraps `data/pokemon-showdown/` with the official server and
`npm install`. External-opponent runs also clone their pinned source revisions
under `data/opponents/`; Foul Play gets a separate Python environment there.
Those directories are covered by the repository's existing `data/` ignore rule.
Later runs reuse both installations. The runner also reuses a server that is
already listening on its configured port.

Prerequisites:

- the existing `outputs/interaction-v1-1epoch/policy/final/` checkpoint;
- the cached Qwen base model used during training;
- the project's Python dependencies;
- Git, Node.js, and npm for the first local-server setup;
- `uv` for Foul Play's isolated `poke-engine` installation;
- CUDA for the checkpoint's default 4-bit loading mode.

## What happens at each decision

Pokémon Showdown sends public battle events and a private request describing
the current legal choice. `poke-env` turns that stream into a `Battle` object.
The live adapter then:

1. converts the player-visible battle into the Metamon-style state schema used
   by the training data;
2. retains all four known move slots, every surviving bench Pokémon, the
   player's request-only Tera types, and simultaneous conditions/effects so
   action identities do not shift when a choice becomes unavailable;
3. maps `Battle.available_moves`, `Battle.available_switches`, forced-switch
   state, and Tera availability onto an exact A0-A12 mask;
4. updates stable roster slots and four backward-looking transition records;
5. builds the interaction tensors without a target action or outcome label;
6. runs Qwen and the interaction head, normalizes the legal-action
   log-probabilities, and selects the highest-preference legal action;
7. maps that action back to a `poke-env` order and writes the complete decision
   trace.

The separation between the full action inventory and exact request mask is
intentional. If Moonblast is disabled, for example, later moves keep their
original alphabetical action IDs rather than moving into Moonblast's slot.

Forced switches, trapping, zero-availability moves, Tera actions, Struggle,
Recharge, and the exact switch list are handled by the live mapping. If state
conversion, inference, or order mapping raises an error, the player logs the
exception and submits a random valid Showdown order so the battle can finish.
Use `--fail-fast` while debugging to stop instead.

## Opponents and teams

Available local opponents are:

```text
random       uniform random valid orders
max-power    highest available base-power move
heuristic    poke-env's switching, setup, hazard, damage, and Tera heuristic
pokechamp-one-step  PokéChamp's immediate turns-to-faint baseline
pokechamp-abyssal  PokéChamp's switching, setup, hazard, and damage heuristic
foul-play     Foul Play's sampled-state MCTS engine
```

The external integrations execute the published policies as separate processes;
they do not reimplement their choice rules in this repository. The exact source
revisions are pinned and written into each summary:

When Foul Play is selected, the report directory also contains
`foul_play_teacher.jsonl`. This is the MCTS policy-distillation dataset from
Foul Play's decisions in those games. It does not change the live evaluation or
load another Qwen model. See [Foul Play policy distillation](foul-play-distillation.md)
for the schema and training command.

| Opponent | Source revision | License | Runtime behavior |
| --- | --- | --- | --- |
| PokéChamp One-Step | `0f84c460319ebe733f8c3028e58a2a5452c60d85` | MIT | Scores immediate attacks by estimated turns to faint; it does not proactively switch in the selected published path. |
| PokéChamp Abyssal | `0f84c460319ebe733f8c3028e58a2a5452c60d85` | MIT | Uses hand-authored matchup, switching, setup, hazard, damage, and Tera rules. |
| Foul Play | `25c976f05cbf2880eaa579afd6db1dcb2c3b57c6` | GPL-3.0 | Samples hidden sets and searches actions with its Rust-backed `poke-engine`; the default is 100 ms, one search worker, and one search thread. |

Run twenty games against each with:

```bash
python -m pokemon_battler.live_eval --games 20 --opponent pokechamp-one-step
python -m pokemon_battler.live_eval --games 20 --opponent pokechamp-abyssal
python -m pokemon_battler.live_eval --games 20 --opponent foul-play
```

Use `--no-bootstrap-opponents` to require the pinned checkouts and isolated
environment to exist already. Foul Play's compute can be changed with
`--foul-play-search-time-ms`, `--foul-play-parallelism`, and
`--foul-play-search-threads`; changing them creates a different benchmark.

The default player and opponent both use
`examples/teams/gen9ou-balance.txt`. Supply different Showdown export files
with:

```bash
python -m pokemon_battler.live_eval \
  --games 100 \
  --opponent heuristic \
  --team-file /path/to/player-team.txt \
  --opponent-team-file /path/to/opponent-team.txt
```

The trained player retains submitted team order and leads with slot one. The
built-in opponents do the same. PokéChamp retains its published randomized team
preview, while Foul Play retains its published search-based preview. Both fields
are recorded in `summary.json`. A serious strength estimate should repeat the
same frozen policy over a versioned pool of teams and leads.

## Fixed-team benchmark on August 9, 2026

The completed checkpoint was tested for 20 games against each new opponent with
the same bundled Gen 9 OU team supplied to both sides. The model used a fixed
slot-one lead. No run used Ollama, an API model, PPO, or additional training.

| Opponent | Record | Win rate | Wilson 95% interval | Decisions | Fallbacks | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PokéChamp One-Step | 20–0 | 100% | 83.9%–100% | 516 | 0 | 21.65 |
| PokéChamp Abyssal | 20–0 | 100% | 83.9%–100% | 612 | 0 | 26.15 |
| Foul Play, 100 ms | 4–16 | 20% | 8.1%–41.6% | 531 | 0 | 21.25 |

The two perfect heuristic records show that the policy can execute complete,
legal battles and exploit relatively local rules. They do not imply a 100%
general win rate: One-Step is especially limited because its selected policy
does not proactively switch. The Foul Play result is the stronger measurement.
At 4–16, the trained policy won real games against a lookahead opponent but was
clearly weaker under these conditions.

Twenty games still produce wide intervals, and this is a single mirrored team
rather than a team-pool or ladder estimate. Do not combine the three records
into one 44–16 score: the opponents have radically different strength. The
zero-fallback result across 1,659 measured decisions is separately useful—it
shows that live state conversion, exact legal masking, and order submission
survived the full benchmark.

## Outputs

Each run gets a new `reports/live/<timestamp>/` directory:

- `run_config.json` records all command arguments;
- `showdown.log` records local server output;
- `opponent.log` records the external opponent when one is selected;
- `opponent.ready` and, for Foul Play, `opponent.start` record lifecycle gates;
- `decisions.jsonl` records every observation, exact legal mask, action
  distribution, selected order, auxiliary value estimate, latency, and
  fallback reason;
- `foul_play_teacher.jsonl`, for Foul Play runs, records its unfiltered MCTS
  visit distribution and the matching public observation for distillation;
- `replays/` contains Showdown replay logs;
- `summary.json` contains wins, losses, ties, the raw win rate, a 95% Wilson
  interval, decision and fallback counts, latency statistics, and per-battle
  results.

The action percentages are policy preferences, not calibrated probabilities
that an action will win. The auxiliary value output is also recorded for
diagnosis but remains uncalibrated and must not be reported as match-win
probability.

## Server controls

Useful options include:

```bash
# Reuse an installation but refuse automatic cloning or npm installation.
python -m pokemon_battler.live_eval --no-bootstrap-server

# Connect to a separately managed local server on another port.
python -m pokemon_battler.live_eval \
  --server-port 9000 \
  --showdown-dir /path/to/pokemon-showdown \
  --no-bootstrap-server

# Leave a server started by the runner alive after evaluation.
python -m pokemon_battler.live_eval --keep-server
```

The local player deliberately keeps connection details separate from policy
logic. Public account authentication, bounded challenge/ladder matchmaking,
trajectory capture, and opt-in between-batch PPO are implemented separately in
`pokemon_battler.public_play`; they reuse this model and state converter. See
[Public Showdown play and between-game learning](public-showdown-learning.md).
