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
`npm install`. That directory is covered by the repository's existing `data/`
ignore rule. Later runs reuse the installation. The runner also reuses a server
that is already listening on its configured port.

Prerequisites:

- the existing `outputs/interaction-v1-1epoch/policy/final/` checkpoint;
- the cached Qwen base model used during training;
- the project's Python dependencies;
- Git, Node.js, and npm for the first local-server setup;
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
```

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

Both sides retain the submitted team order and lead with slot one. This makes
the initial comparison reproducible, but it also means the result includes a
deliberately simple, untrained lead policy. A serious estimate should repeat the
same frozen policy over a versioned pool of teams and leads.

## Outputs

Each run gets a new `reports/live/<timestamp>/` directory:

- `run_config.json` records all command arguments;
- `showdown.log` records local server output;
- `decisions.jsonl` records every observation, exact legal mask, action
  distribution, selected order, auxiliary value estimate, latency, and
  fallback reason;
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
logic. Public play later replaces the server configuration and calls ladder or
challenge matchmaking; it does not require another model or state converter.
