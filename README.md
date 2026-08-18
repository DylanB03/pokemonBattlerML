# Pokémon Battler

I trained a `Qwen/Qwen2.5-0.5B` Pokémon policy to play Generation 9 OU. The
strongest checkpoint finished **502-498 over 1,000 public Pokémon Showdown
games**, a **50.2% win rate**.

## Result

| Public ladder measurement | Value |
| --- | ---: |
| Checkpoint | `outputs/metamon-large-v2/04-candidate` |
| Games | 1,000 |
| Wins | 502 |
| Losses | 498 |
| Win rate | **50.2%** |
| 95% Wilson interval | 47.1%-53.3% |
| Opponents | 770 |
| Policy decisions | 30,385 |
| Policy fallbacks | 0 |
| Mean battle length | 24.8 turns |

The first 100-game trace finished 53-47. A later trace using the same frozen
checkpoint and deployment settings finished 449-451 over 900 games. The traces
share no battle IDs, so the table reports all 1,000 completed games rather than
the more favorable first slice.

The Showdown account `ATSskipper5` started at **1000 ELO** and showed **1189
ELO** on August 18, 2026. The account also contains earlier policies, cancelled
runs, and disconnect losses, so I cannot assign the full 189-point increase to
this checkpoint. The controlled result is the 502-498 record. Its confidence
interval still includes 50%, so the four-game margin is positive without being
conclusive evidence that the underlying win rate is above even.

## The final policy

The deployed policy combines Qwen with a learned structured sidecar:

```text
public battle state ──> Qwen interaction policy ───────> legal log probabilities
                   └─> structured mechanics sidecar ──> legal log probabilities
Showdown request ─────> exact legal-action mask

                       Qwen + 0.75 × sidecar ──> greedy action
```

Qwen remains part of every live decision. The sidecar adds a second learned
distribution over the same 13 action slots using:

- numeric move, matchup, damage-pressure, hazard, status, and switch features;
- categorical move, species, item, ability, type, field, and effect identities;
- stable player and opponent rosters;
- four previous observable battle transitions; and
- the exact legal-action mask supplied by Showdown.

The public measurement used a fixed submitted team, a randomized team-preview
lead, greedy actions, and up to four simultaneous battles. The checkpoint did
not learn between those games. No MCTS, Foul Play, `poke-engine`, or other
search policy selected moves at inference time.

Model and data outputs are ignored by Git. The checkpoint path above refers to
the trained local artifact, not weights stored in this repository.

## What improved the model

Most of the project's experiments did not improve full-game performance. These
were the changes that survived measurement.

### 1. Train the decision directly

The first objective trained Qwen to generate strings such as `A4`. Its loss
also rewarded easy `A` and EOS tokens, which made a low loss look better than
the resulting decisions. A directly masked action loss trained one
cross-entropy over the legal candidates instead.

The candidate-head reference reached 36.52% exact agreement on a fixed
1,024-row validation sample. This also separated legal-action enforcement from
action quality: the mask removes illegal choices, while the model must still
rank the remaining choices.

### 2. Put battle mechanics into the input

The early candidate model had to recover type, power, status effects, stat
changes, and switch costs from names. The mechanics-v2 representation supplied
those values directly and retained categorical identities for actions that
cannot be reduced to the same numeric flag.

On the same fixed 1,024 rows, exact action agreement rose from **36.52% to
41.89%**. The completed 5,000-row evaluation reached 42.86% top-1, 64.78%
top-2, and 78.78% top-3 agreement.

### 3. Change the amount and variety of training data

The earlier Foul Play collections had about 51,000 turn labels from 1,000
battles and six training opponent teams. The first large Metamon run retained
34,524 trajectories and 1,249,105 decisions from a deterministic 0.5% slice of
the published self-play archives.

That v1 sidecar finished **48-52** on a paired 100-game held-out schedule while
the previous champion finished **30-70** on the same schedule. This was the
first battle result large enough to continue the policy rather than replace it
with another architecture.

### 4. Continue the sidecar instead of restarting it

The v2 run used the next disjoint 0.5% hash window. It retained 34,735
trajectories and 1,259,031 transitions, loaded the v1 sidecar, and mixed enough
v1 cache rows into training for a 25% rehearsal share. A battle sweep selected
the 0.75 sidecar blend.

On a separate 200-game held-out schedule, v2 finished **109-91** while v1
finished **96-104**. I then froze v2 for the 1,000-game public measurement.

Public PPO, small teacher-distillation sets, the recurrent IQL policy, and the
conservative residual did not produce a measured improvement over the selected
champion. Their implementations remain in the repository for reproducibility,
but they are not part of the final training path.

## Install

Run commands from the repository root in an active Python 3.10 or newer
environment:

```bash
uv pip install -e .
python -m unittest discover -s tests -v
```

The default training configuration expects an NVIDIA GPU for 4-bit QLoRA.
Local Showdown evaluation also requires Git, Node.js, and npm. The large
Metamon pipeline downloads and prepares tens of gigabytes of data; its storage
guards stop prepared output above 32 GiB and cache construction above 16 GiB
unless explicitly changed.

## Reproduce the final training path

Train the first large-data sidecar:

```bash
python -m pokemon_battler.large_offline_pipeline \
  --output-dir outputs/metamon-large-v1
```

Continue it on the next disjoint data slice:

```bash
python -m pokemon_battler.large_offline_pipeline \
  --output-dir outputs/metamon-large-v2 \
  --checkpoint outputs/metamon-large-v1/04-candidate \
  --trajectory-sample-rate 0.005 \
  --trajectory-sample-offset 0.005 \
  --rehearsal-run-dir outputs/metamon-large-v1 \
  --rehearsal-ratio 0.25 \
  --learning-rate 3e-5 \
  --epochs 3 \
  --blend-sweep-games 50 \
  --blend-sweep-weight 0.25 \
  --blend-sweep-weight 0.5 \
  --blend-sweep-weight 0.75 \
  --blend-sweep-weight 1.0 \
  --games 200 \
  --minimum-delta-interval-lower 0
```

The strict automatic promotion interval remained on v1 because the paired
interval crossed zero. The v2 candidate had the stronger point estimate in
both held-out battle stages and is the checkpoint used for the public result:

```text
outputs/metamon-large-v2/04-candidate
```

The pipeline streams compressed archives, resumes completed preparation and
cache shards, uses four CPU workers where useful, and loads only one Qwen copy
on the GPU. See [Large offline self-play](docs/large-offline-selfplay.md) for
data sources, storage, action-parity checks, artifact layout, and the full
training objective.

## Evaluate locally

Run the final checkpoint against the local heuristic:

```bash
python -m pokemon_battler.live_eval \
  --checkpoint outputs/metamon-large-v2/04-candidate \
  --games 20 \
  --opponent heuristic \
  --team-preview-policy random
```

Available opponents include `random`, `max-power`, `heuristic`,
`pokechamp-one-step`, `pokechamp-abyssal`, and `foul-play`. The local
runner installs and starts the official Showdown server when necessary and
writes summaries, decisions, and replays under `reports/live/`.

## Play on public Showdown

Copy the environment template and add the registered bot account:

```bash
cp .env.example .env
python -m pokemon_battler.public_play --mode login
```

Run a frozen 1,000-game ladder measurement:

```bash
python -m pokemon_battler.public_play \
  --mode ladder \
  --checkpoint outputs/metamon-large-v2/04-candidate \
  --games 1000 \
  --batches 1 \
  --concurrent-games 4 \
  --team-preview random \
  --output-dir reports/public/metamon-large-v2-frozen-1000-001
```

If the process is interrupted, rerun the same command with `--resume`.
Completed battle IDs remain in the append-only trace and are not requested
again.

`poke-env` can leave a ladder search waiting forever when Showdown does not
send the expected battle-start notification. The current runner stays
connected but stops making progress. Stop it with `Ctrl+C` and resume the
same output directory. A watchdog that retries stale searches without
interrupting active battles is still needed.

Public ladder automation is subject to Pokémon Showdown's rules, rate limits,
bot restrictions, and staff discretion.

## Documentation

The repository has several documents because I kept the design and result of
each major experiment instead of rewriting the history after it failed. Only
the first four are needed to understand or run the final policy.

| Current document | Why it exists |
| --- | --- |
| [Training journey](docs/training-journey.md) | Full first-person account from the original SFT loss through the 1,000-game result |
| [Large offline self-play](docs/large-offline-selfplay.md) | Final v1 and v2 data, training, continuation, and evaluation pipeline |
| [Public Showdown](docs/public-showdown-learning.md) | Account setup, frozen ladder runs, resume behavior, summaries, and known matchmaking stall |
| [Local Showdown](docs/live-showdown-evaluation.md) | Local server setup, opponents, state conversion, and replay artifacts |

Two technical references describe inputs inherited by the final sidecar:

| Technical reference | Relevance |
| --- | --- |
| [Mechanics v2](docs/mechanics-v2.md) | Numeric and categorical candidate features |
| [Interaction policy v3](docs/interaction-policy-v3.md) | Stable rosters, history events, cache schema, and interaction tokens |

The remaining documents are experiment records. They are not recommended
training instructions:

| Historical document | Result |
| --- | --- |
| [Mechanics v1](docs/mechanics-v1.md) | Legacy numeric schema replaced after exact feature collisions |
| [Mechanics-v2 results](docs/mechanics-v2-results-and-next-steps.md) | Postmortem that led to interaction tokens and value heads |
| [Qwen win training](docs/qwen-win-training.md) | Small offline-RL and PPO path later beaten by the Metamon sidecar |
| [Foul Play distillation](docs/foul-play-distillation.md) | Search-teacher collection with too little team coverage |
| [Gated improvement](docs/gated-improvement.md) | Bounded teacher experiment that did not prove a battle gain |
| [Trajectory IQL](docs/trajectory-iql.md) | Recurrent policy that lost its architecture and champion gates |
| [Champion residual](docs/champion-residual.md) | Conservative residual whose final held-out battle test tied the champion |

[ROADMAP.md](ROADMAP.md) describes possible work beyond the current battler,
including calibrated action values, replay review, counterfactual analysis, and
coaching.

## Limits

- The public result used one fixed player team. It measures that deployed setup,
  not broad team-building ability.
- Team preview was randomized because the selected checkpoint has no learned
  preview head.
- The structured action-value targets come from logged final outcomes. They do
  not reveal what would have happened after unplayed actions.
- The public account rating mixes several policies and operational failures.
- The checkpoint uses a 0.5B language model and has no inference-time search.
- The 50.2% point estimate is close enough to even that another independent
  1,000-game set could finish below 50%.

## Data and attribution

The large pipeline uses the
[Metamon self-play corpora](https://github.com/UT-Austin-RPL/metamon). The
published Hugging Face dataset is marked `CC-BY-NC-4.0`; review its current
license and attribution requirements before publishing checkpoints, derived
data, or a hosted service.

The model uses [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), and
complete battles run through [Pokémon Showdown](https://pokemonshowdown.com/).
