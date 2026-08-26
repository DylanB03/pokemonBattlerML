# Pokémon Battler

I trained a `Qwen/Qwen2.5-0.5B` policy to play Generation 9 OU. The selected
checkpoint finished **502-498 across 1,000 public Pokémon Showdown games**, a
**50.2% win rate**, while making every decision with the learned policy.

| Public ladder result | Value |
| --- | ---: |
| Games | 1,000 |
| Record | **502-498** |
| Win rate | **50.2%** |
| Opponents | 770 |
| Policy decisions | 30,385 |
| Rule-based fallbacks | 0 |
| Mean battle length | 24.8 turns |

The measured checkpoint is `outputs/metamon-large-v2/04-candidate`. Model and
dataset outputs are ignored by Git, so the trained weights are not included in
this repository.

[Read the full project write-up](https://www.dylanb.ca/projects/pokemon-battler)
for the decisions, experiments, results, and limitations behind the final
system.

## How the policy works

The deployed policy combines two learned action rankings:

```text
compact battle state ──> Qwen2.5-0.5B ─────────> legal-action scores
mechanics + identities ─> structured sidecar ──> legal-action scores

                     Qwen + 0.75 × sidecar ──> highest-scoring legal action
```

Qwen reads a compact description of the public battle state. The sidecar reads
numeric mechanics, categorical identities, both visible rosters, and four
recent public transitions. Pokémon Showdown supplies the exact legal-action
mask; it does not recommend a move. The live policy does not use MCTS, a damage
engine, Foul Play, or another search policy at inference time.

## Install

Use Python 3.10 or newer from the repository root:

```bash
uv pip install -e .
python -m unittest discover -s tests -v
```

Training expects an NVIDIA GPU for 4-bit QLoRA. Local Showdown evaluation also
requires Git, Node.js, and npm.

## Train the selected policy

The first command downloads a deterministic 0.5% sample of the Metamon
self-play corpus, prepares it, builds feature caches, trains the structured
sidecar, and evaluates the candidate:

```bash
pokemon-large-offline-run \
  --output-dir outputs/metamon-large-v1
```

The second command continues that sidecar on a disjoint 0.5% sample and mixes
in 25% rehearsal data from the first run:

```bash
pokemon-large-offline-run \
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

The two completed runs retained 69,259 battles and about 2.5 million decisions
in total. They consumed roughly 72 GiB under `outputs/` on the development
machine. Preparation is resumable, but reproducing both stages needs enough
space for source archives, prepared rows, numeric caches, and checkpoints.

See [Large offline self-play](docs/large-offline-selfplay.md) for the data
layout, resume behavior, storage guards, and evaluation stages.

## Evaluate locally

Run the selected checkpoint against a local heuristic opponent:

```bash
pokemon-live-eval \
  --checkpoint outputs/metamon-large-v2/04-candidate \
  --games 20 \
  --opponent heuristic \
  --team-preview-policy random
```

The local runner supports `random`, `max-power`, `heuristic`,
`pokechamp-one-step`, `pokechamp-abyssal`, and `foul-play`. It writes summaries,
decision traces, and replays under `reports/live/`.

## Play on public Showdown

Copy the environment template, add the registered bot account, and verify the
login:

```bash
cp .env.example .env
pokemon-public-play --mode login
```

Run a frozen 1,000-game ladder measurement:

```bash
pokemon-public-play \
  --mode ladder \
  --checkpoint outputs/metamon-large-v2/04-candidate \
  --games 1000 \
  --batches 1 \
  --concurrent-games 4 \
  --team-preview random \
  --output-dir reports/public/metamon-large-v2-frozen-1000-001
```

Rerun the same command with `--resume` after an interruption. Completed battle
IDs remain in the append-only trace. Public automation remains subject to
Pokémon Showdown's rules, rate limits, bot restrictions, and staff discretion.

## Source layout

```text
src/pokemon_battler/
├── core/        action encoding, observations, prompts, and mechanics
├── data/        replay preparation, datasets, caches, and team manifests
├── models/      Qwen heads and structured policy networks
├── training/    supervised, outcome-aware, distillation, and RL trainers
├── evaluation/  metrics, baselines, ablations, and policy suites
├── showdown/    local server, public ladder, opponents, and live state
└── pipelines/   end-to-end experiment orchestration
```

Installed command names remain stable even though their implementation now
lives in these subpackages.

## Technical documentation

- [Large offline self-play](docs/large-offline-selfplay.md): final dataset,
  cache, continuation, and evaluation pipeline.
- [Public Showdown play](docs/public-showdown-learning.md): account setup,
  frozen and learning runs, resume behavior, and report files.
- [Local Showdown evaluation](docs/live-showdown-evaluation.md): local server,
  opponents, state conversion, and replay artifacts.
- [Mechanics v2](docs/mechanics-v2.md): numeric and categorical action features.
- [Interaction policy v3](docs/interaction-policy-v3.md): roster, history, cache,
  and interaction-token schemas.

The development narrative lives in the
[project write-up](https://www.dylanb.ca/projects/pokemon-battler), rather than
being duplicated across the README and repository notes.

## Data and attribution

The large pipeline uses the
[Metamon self-play corpus](https://github.com/UT-Austin-RPL/metamon), published
as `CC-BY-NC-4.0`. Review its current license before publishing checkpoints,
derived data, or a hosted service.

The base model is
[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), and battles run through
[Pokémon Showdown](https://pokemonshowdown.com/).
