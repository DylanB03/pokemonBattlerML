# Pokémon Battler Policy-Training Harness

This repository fine-tunes a small causal language model to choose actions in
Generation 9 OU Pokémon battles. It converts Metamon replay trajectories into
turn-level examples, trains `Qwen/Qwen2.5-0.5B` with QLoRA, and evaluates only
actions that were legal in each recorded state.

The original objective is supervised behavior cloning: predict a strong human
player's recorded action from the visible battle state. The win-training path
now uses that checkpoint as a warm start, fits outcome-aware action and state
values, and then optimizes Qwen directly through self-play PPO. Human-action
agreement remains diagnostic; promotion is decided by complete-game results.

## Train on a dataset large enough to change the experiment

The recommended next run keeps the batch-005 Qwen policy and adds a structured
sidecar trained on the official Metamon `pac-base` and `pac-exploratory`
self-play corpora:

```bash
python -m pokemon_battler.large_offline_pipeline \
  --output-dir outputs/metamon-large-v1
```

The command downloads the two Gen 9 OU archives and the current general/high
ladder team sets, prepares trajectory shards with four worker processes, builds
the numeric interaction caches four at a time, trains with four data-loader
workers, and evaluates four local games concurrently. Completed shards and
caches are reused after an interruption. The default 5% deterministic sample is
already much broader than the six-team teacher runs; use
`--trajectory-sample-rate 1` only if there is enough disk for the full Gen 9
portion of both archives.

This does not run four copies of Qwen on one GPU. The sidecar learns from the
same mechanics, roster, candidate, and four-event history tensors already used
by the Qwen interaction head, while Qwen remains the base distribution at live
inference. Their legal-action log probabilities are blended. That makes a
million-row run practical without pretending the language model can be encoded
millions of times cheaply. Every output is a new checkpoint, and the held-out
battle gate leaves `selected_checkpoint.txt` on the old champion when the
candidate loses.

See [Large offline self-play pipeline](docs/large-offline-selfplay.md) for the
data sources, action-parity check, resume behavior, throughput design, disk
tradeoffs, artifact layout, and limitations. The recurrent IQL and conservative
residual experiments remain reproducible, but both completed battle evaluations
were negative and neither is the recommended next run.

See [ROADMAP.md](ROADMAP.md) for the planned progression from behavior cloning
to model-preference displays, replay review, counterfactual win-probability
analysis, regret estimation, and grounded coaching.

For the development story behind the current objective and experiment design,
read [I Was Training a Pokémon Policy to Write A4](docs/training-journey.md).

## Train Qwen directly for wins

The complete pilot command is:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-pilot-1
```

It reuses or prepares the replay dataset, performs an outcome-conditioned
offline warm start, collects sampled Qwen-versus-frozen-Qwen games on the local
official Showdown server, runs clipped PPO updates, and promotes candidates
through deterministic win-rate matches. Showdown executes game rules and
returns observations; it never supplies an action recommendation. No MCTS,
`poke-engine`, Foul Play, or external policy participates in training or
inference.

Every offline, candidate, rejected, and promoted model is stored separately.
See [Training Qwen for battle wins](docs/qwen-win-training.md) for the objective,
default budget, league behavior, team-pool controls, artifacts, limitations,
and smoke-test command.

## Run the trained policy in local Showdown battles

The completed interaction checkpoint can now play full Generation 9 OU battles
without another training run. From the repository root:

```bash
python -m pokemon_battler.live_eval --games 20 --opponent heuristic
```

On its first run, the command clones the official Pokémon Showdown server into
the ignored `data/pokemon-showdown/` directory, installs its Node dependencies,
starts it locally with `--no-security`, loads the existing `final/` checkpoint
once, and plays the bundled team against a deterministic-preview
`SimpleHeuristicsPlayer`. It requires `git`, Node.js, and `npm`. If a server is
already listening on port 8000, the runner reuses it.

Every decision uses Showdown's exact request mask while retaining the same
alphabetical A0-A12 action identities used by the Metamon training data. Results
are written under `reports/live/<timestamp>/`:

```text
summary.json       win rate, Wilson interval, fallbacks, and inference latency
decisions.jsonl    state, legal mask, complete preference distribution, and order
replays/           Pokémon Showdown replay logs
showdown.log       managed local-server output
```

Use `--opponent random`, `--opponent max-power`, `--opponent heuristic`,
`--opponent pokechamp-one-step`, `--opponent pokechamp-abyssal`, or
`--opponent foul-play`. The three external policies run at pinned upstream
revisions in isolated processes; no Ollama model is involved. The bundled team
is a fixed-team fixture, not a broad team-pool evaluation. See
[Local Showdown evaluation](docs/live-showdown-evaluation.md) for setup, state
conversion, exact benchmark results, failure handling, and CLI controls.

Foul Play can also act as an offline MCTS teacher. `pokemon-teacher-collect`
now runs fixed-team Foul Play against a second Foul Play agent while cycling the
opponent through a randomized, prevalidated OU team pool. This keeps both the
soft labels and visited-state distribution search-backed. It prints progress
after every game and fails stalled or illegal-team collections explicitly.
`pokemon-distill` then fits a new Qwen interaction checkpoint without
overwriting the source model. See
[Foul Play policy distillation](docs/foul-play-distillation.md).

The old cumulative `pokemon-win-pipeline` is retained for reproducibility, but
it is no longer the recommended training path. It repeatedly recomputes frozen
Qwen states, grows its dataset every round, and combines preview and Q-value
changes before proving that either helps in battles.

Use the gated pipeline instead:

```bash
python -m pokemon_battler.gated_pipeline \
  --output-dir outputs/qwen-gated-v1
```

The defaults point at the completed batch-005 champion, the round-00 distilled
candidate, both existing smart-vs-smart teacher collections, the current replay
split, and the bundled enemy-team manifest. The command first measures preview
and Q-blend inference choices independently. It then collects a held-out Foul
Play ceiling/validation set, selects 8,000 high-information teacher states and
bounded replay samples, runs frozen Qwen exactly once per selected state, and
trains three cheap head objectives from the same cache. A memorization gate and
an unseen-team/replay gate can stop the run before battle evaluation. Only the
best eligible head receives a 100-game paired test.

No checkpoint is overwritten. `selected_checkpoint.txt` points to the promoted
model or back to the starting champion, while every rejected model and every
gate report remains under the run directory. Foul Play and Showdown are used
only for local collection and evaluation; deployment still loads Qwen and the
learned PyTorch head. See [Gated Qwen improvement](docs/gated-improvement.md)
for the rationale, stages, estimates, overrides, and artifact layout.

On the completed 20-game fixed-team samples, the checkpoint scored 20–0 against
PokéChamp One-Step, 20–0 against PokéChamp Abyssal, and 4–16 against Foul Play's
100 ms search. All 1,659 decisions completed without a fallback. These results
show reliable full-battle execution and a clear gap to the search opponent; they
are not a ladder or multi-team win-rate estimate.

## Play on public Showdown and optionally learn between batches

Public play uses a normal registered Pokémon Showdown account loaded from an
ignored `.env`; credentials never enter run artifacts. Start by copying
`.env.example`, filling in `POKEMON_SHOWDOWN_USERNAME` and
`POKEMON_SHOWDOWN_PASSWORD`, and verifying the login without loading Qwen:

```bash
cp .env.example .env
python -m pokemon_battler.public_play --mode login
```

The recommended first human test accepts a fixed account's unrated challenges:

```bash
python -m pokemon_battler.public_play \
  --mode accept \
  --opponent YourTestingAccount \
  --games 20
```

Add `--learn --batches 3 --games 32` only after the frozen run is clean. The
policy stays frozen within every public batch. Between batches, PPO writes a
new candidate, local candidate-versus-champion games decide promotion, and all
accepted and rejected checkpoints remain separate. The default public policy
comes from `outputs/qwen-win-pilot-1/selected_checkpoint.txt`, not the older
interaction-policy default. Finite public runs have no overall wall-clock
timeout and wait for all requested games to finish. The terminal reports every
completed game's result, cumulative win-loss-tie record, win rate, and exact
rated ELO transition. Every batch then prints its starting and ending ELO, net
change, points gained and lost, peak and low alongside the compact public-batch,
PPO, and promotion summaries. The campaign JSON retains the same rating metrics
per 100-game suite and overall. See
[Public Showdown play and between-game learning](docs/public-showdown-learning.md)
for `.env` fields, commands, safeguards, artifacts, and ladder limitations.
The documented bounded campaign mode can run ten 100-game learning batches,
chain each promoted PPO model from its predecessor, stop early above a 50%
batch score, and emit per-batch plus aggregate model-improvement reports.

## Run the new interaction policy end to end

From the repository root, this is the complete command:

```bash
python -m pokemon_battler.interaction_experiment
```

It performs the entire experiment in order:

1. scans `data/raw/metamon/gen9ou.tar.gz` and prepares a deterministic 2%
   schema-3 train/validation/test split;
2. builds strict memory-mapped interaction caches for train and validation;
3. trains on 128 rows and requires at least 95% training accuracy as a wiring
   gate;
4. trains the full hierarchical interaction policy for up to one epoch;
5. evaluates `best/` and `final/` on the same deterministic validation sample;
6. writes a combined summary without opening the test split.

The first run must scan the roughly 20.4 GB compressed archive, so preparation
and cache construction happen before GPU training begins. Matching prepared
data and caches are reused on later experiments. The default expects the Qwen
checkpoint to be cached locally and an NVIDIA GPU capable of 4-bit QLoRA; add
`--allow-download` only if the checkpoint is not already local.

Artifacts are kept separately:

```text
data/gen9ou-interaction-v1/
outputs/interaction-v1-1epoch/overfit-128/
outputs/interaction-v1-1epoch/policy/best/
outputs/interaction-v1-1epoch/policy/final/
outputs/interaction-v1-1epoch/policy/run_summary.json
outputs/interaction-v1-1epoch/end_to_end_summary.json
```

The runner refuses a nonempty output directory, which prevents an older
post-trained model from being overwritten. Select a different `--output-dir`
for every later run. After an editable reinstall, `pokemon-interaction-run` is
the equivalent entry point. The implemented tensors and model contract are in
[Interaction policy v3](docs/interaction-policy-v3.md).

## Mechanics-v2 result and why the architecture changed

The completed mechanics-v2 run selected `final/` at 42.86% exact agreement on
5,000 validation rows, with 64.78% top-2 and 78.78% top-3 agreement. On the
fixed 1,024-row training-validation sample, its best checkpoint reached 41.89%,
up from the candidate-head reference of 36.52% with the same 0.5B base model.

The existing `data/gen9ou-dev` files were prepared before the current schema
and lack multi-turn move history, accumulated opponent reveals, and legal-mask
provenance. The interaction runner now regenerates those rows, repeats a
memorization gate, and trains once on the corrected data. The prior result,
capacity analysis, and rationale for changing the architecture are in
[What mechanics-v2 proved, and what should change next](docs/mechanics-v2-results-and-next-steps.md).
The tensor shapes, token layout, history rules, hierarchical loss, and
value-head boundary are specified in
[Interaction policy v3](docs/interaction-policy-v3.md).

## Reproduce the completed experiment

With the existing `data/gen9ou-dev` split and cached Qwen checkpoint, this one
command trains for up to one complete dataset pass, evaluates `best/` and `final/` on
the same 5,000 hash-sampled validation rows, and writes a comparison summary:

```bash
python -m pokemon_battler.experiment \
  --output-dir outputs/mechanics-v2-1epoch
```

The defaults are candidate-wise legal-action cross-entropy, a shared numeric
and categorical mechanics scorer, a compact state prompt, QLoRA, effective
batch 32, SDPA, a cosine schedule with a 5% floor, and seed 42. Exact move and
state identities use learned embeddings rather than arbitrary ordinal numbers.
The runner builds memory-mapped numeric feature caches before training. No
`--max-steps` is used, so the 286,059-row training split allows about 8,940
optimizer updates. Early stopping ends the run after four 500-step validation
checks without an accuracy gain of at least 0.002. Results are saved to:

```text
outputs/mechanics-v2-1epoch/best/
outputs/mechanics-v2-1epoch/final/
outputs/mechanics-v2-1epoch/history.json
outputs/mechanics-v2-1epoch/reports/best-validation.json
outputs/mechanics-v2-1epoch/reports/final-validation.json
outputs/mechanics-v2-1epoch/run_summary.json
```

The first cache build is CPU work and writes about 1.43 GiB for training and
147 MiB for validation. Later runs reuse those files when the source JSONL and
feature schema are unchanged. `mechanics-v1` caches and checkpoints remain
loadable, but they are not reused by a v2 run.

This command requires CUDA for its default 4-bit configuration. After an
editable reinstall, `pokemon-run --output-dir ...` is the equivalent entry
point.

## Experiment protocol

This is the default protocol for one 8 GB NVIDIA GPU:

| Phase | Data and budget | Purpose | Required result |
| --- | --- | --- | --- |
| Environment check | Unit tests and cached model | Verify the installation | All tests pass and CUDA is visible |
| Data development set | 2% deterministic turn sample | Retain battle diversity at manageable cost | At least 50k train and 5k validation rows |
| Memorization gate | 128 rows, at most 400 updates | Detect broken labels, masking, LoRA, or loading | At least 95% accuracy on the same 128 rows |
| First model | Up to one complete pass, about 8,940 updates | Test the mechanics-conditioned objective | Best hash-sampled validation action accuracy |
| Main data | 10% deterministic turn sample | Scale only after the first result is sound | Reinspect lengths before training |
| Main policy | One to three complete passes | Produce the selected checkpoint | Compare `best/` and `final/` |
| Final offline test | 5,000 held-out test rows | Report imitation performance once | Compare base and trained policies |
| Battle evaluation | Fixed teams and opponents | Measure actual policy quality | Win rate with confidence intervals |

The budgets are optimizer updates, not guesses based on an archive's byte size.
The current local `gen9ou.tar.gz` is about 20.4 GB of compressed trajectories.
That is not 20.4 GB of model tokens, and it does not tell us how much work one
epoch contains.

The recommended model configuration is:

```text
base checkpoint     Qwen/Qwen2.5-0.5B
base weight storage 4-bit NF4 with double quantization
compute dtype       BF16 when supported, otherwise FP16
trainable weights   LoRA on all attention and MLP projections
effective batch     32 examples
context ceiling     4096 tokens, with dynamic per-batch padding
```

Use the official Qwen checkpoint with `--load-in-4bit`. Do not switch to an
Unsloth prequantized checkpoint for this protocol: the current loader explicitly
prepares QLoRA from the CLI flag, and merely selecting an Unsloth checkpoint
would not enable Unsloth's training kernels.

The CLI represents QLoRA with two separate arguments:

| Arguments | Training mode |
| --- | --- |
| `--method lora --load-in-4bit` | **QLoRA: the recommended 8 GB configuration** |
| `--method lora` without `--load-in-4bit` | LoRA over a BF16/FP16 base |
| `--method full` without `--load-in-4bit` | Full-parameter fine-tuning |

`--method lora` describes which parameters are trainable; `--load-in-4bit`
describes how the frozen base weights are stored. Every recommended training
command in this guide includes both arguments and therefore performs QLoRA.

## What is implemented

- Streaming readers for Metamon `.json`, `.json.lz4`, `.tar`, and `.tar.gz`
- Battle-grouped hash splits and chronological splits
- Metamon-compatible alphabetical action ordering
- Removal of missing, terminal, and unrecoverably illegal replay labels
- Dynamic switch options as Pokémon faint
- Assistant-only causal-LM loss on the action ID and EOS tokens
- A directly legal-masked 13-way policy-head objective that avoids action-label
  tokenization bias
- A shared candidate head that scores each legal action description with the
  same network and randomizes candidate order during training
- A 207-value mechanics vector for every candidate, including typed side
  conditions, speed/order context, matchup, corrected damage pressure, move
  effects, stat changes, and switch entry cost
- Thirty-two categorical identity fields with shared learned embeddings for
  moves, species, items, abilities, types, statuses, effects, field state, and
  recent move history
- A hybrid mechanics head that fuses the structured candidates with one Qwen
  state embedding while keeping the prompt compact
- A mechanics-only MLP baseline using the identical structured inputs and legal mask
- Versioned, memory-mapped feature caches that are reused across runs
- A compact prompt whose measured median is 1,157 tokens instead of 2,057
- Validation action NLL, top-k accuracy, MRR, action-family metrics, entropy,
  prediction histograms, and best-checkpoint selection by action accuracy
- Full-epoch scheduler semantics, a configurable LR floor, gradient diagnostics,
  examples/tokens per second, and dataset-pass reporting
- Optional capped class/family weighting
- A single-command train/evaluate/report experiment runner
- Validation-based plateau stopping that preserves the best checkpoint
- Preparation-time turn/player counts, past opponent reveals without future
  leakage, and conservative zero-PP mask refinement
- Schema-3 trajectory rows with stable six-per-side rosters and four strictly
  backward-looking transition events
- A strict multi-array `interaction-v1` cache with schema fingerprints and
  source-signature validation
- A 30-token interaction transformer over global state, both teams, recent
  history, and all legal candidates, with explicit switch-to-roster links
- A normalized move/switch/Tera hierarchy and an auxiliary battle-outcome value
  head with saved log-loss, Brier-score, and threshold-accuracy metrics
- A 128-row memorization gate and one-command raw-data-to-report interaction run
- Full fine-tuning, LoRA, and 4-bit QLoRA loading
- Best-checkpoint selection by validation action accuracy for action heads, with
  action NLL as the tiebreaker; legacy SFT still uses token loss
- Memory-bounded constrained evaluation that cannot select an illegal replay
  candidate
- Reproducible hash-sampled evaluation with uniform, training-frequency, top-k,
  reciprocal-rank, action-type, and oracle-type metrics
- A compact feature-hashed action network as a non-language-model baseline

`prompt.md` is a human-readable example. The authoritative serializer is
`src/pokemon_battler/prompting.py`.

## Policy objectives and baselines

The original `--objective sft` path trains Qwen to generate strings such as
`A4`, then compares complete legal action strings during inference.

The `--objective policy-head` path reads the final prompt representation and
produces exactly 13 logits. Before cross-entropy or prediction, every entry
outside `legal_action_ids` is set to negative infinity. Illegal replay candidates
therefore receive zero probability and cannot be selected. This also removes the
different Qwen token lengths of `A0` and `A10` from the learning objective.

The former `--objective candidate-head` path records the hidden state at
each legal-action description and applies one shared MLP scorer to all of them.
It uses the same legal mask, but makes the score depend directly on the candidate
semantics instead of a permanently assigned A0-A12 output weight. Candidate
order is randomized during training to reduce position shortcuts.

The recommended `--objective mechanics-head` path builds a 207-value vector and
32 categorical identity fields for each A0-A12 candidate. The numeric branch
contains action kind, type effectiveness, corrected Terastallization STAB,
accuracy, PP, priority, approximate damage, speed/order context, typed side
conditions, secondary effects, healing, recoil, stat changes, hazard
interaction, switch HP, and matchup pressure. The categorical branch keeps
Rest distinct from Sleep Talk, Reflect distinct from Light Screen, and other
mechanically special actions distinct even when their coarse numeric summaries
match. It also embeds species, items, abilities, statuses, active effects, and
move history. These are categorical IDs, not continuous numbers.

The v2 prompt retains compact move and Pokémon names because names provide a
cheap residual signal for mechanics that cannot be safely compressed into one
generic flag. It still omits verbose move-stat prose and legal-action
descriptions. Qwen encodes the strategic battle state once; the head scores
each candidate by combining that state embedding with its structured features.
Illegal rows are masked before cross-entropy and argmax.

This is not RAG and does not ask a 0.5B model to parse generated damage prose.
Damage values are reproducible estimates rather than exact simulator rolls
because the replay rows do not carry EVs, IVs, natures, or every transient
modifier. See [docs/mechanics-v2.md](docs/mechanics-v2.md) for the schema,
compatibility rules, and limitations. The original
[mechanics-v1 design](docs/mechanics-v1.md) is retained for reproducibility.

A direct training command equivalent to the training phase of `pokemon-run` is:

```bash
pokemon-train \
  --model Qwen/Qwen2.5-0.5B \
  --train-file data/gen9ou-dev/train.jsonl \
  --validation-file data/gen9ou-dev/validation.jsonl \
  --output-dir outputs/mechanics-v2-1epoch \
  --objective mechanics-head \
  --prompt-format mechanics-v2 \
  --train-mechanics-cache data/gen9ou-dev/train.mechanics-v2.npy \
  --validation-mechanics-cache data/gen9ou-dev/validation.mechanics-v2.npy \
  --method lora \
  --local-files-only \
  --load-in-4bit \
  --dtype auto \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 1e-4 \
  --epochs 1 \
  --min-lr-ratio 0.05 \
  --validation-examples 1024 \
  --validation-sample-mode hash \
  --validation-sample-seed 42 \
  --eval-steps 500 \
  --early-stopping-patience 4 \
  --early-stopping-min-delta 0.002
```

When calling `pokemon-train` directly, create the caches first:

```bash
pokemon-mechanics-cache --data-file data/gen9ou-dev/train.jsonl
pokemon-mechanics-cache --data-file data/gen9ou-dev/validation.jsonl
```

The one-command `pokemon-run` path performs those steps automatically.

`pokemon-evaluate` automatically detects a saved policy head. To evaluate the
unfine-tuned base model with the existing legal-candidate mask on a deterministic
sample of any size, omit `--adapter`:

```bash
pokemon-evaluate \
  --model Qwen/Qwen2.5-0.5B \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 1000 \
  --sample-mode hash \
  --sample-seed 42 \
  --baseline-train-file data/gen9ou-dev/train.jsonl \
  --local-files-only \
  --load-in-4bit \
  --output reports/base-validation-1000.json
```

Use the identical row count, sample mode, and seed for every checkpoint. Reports
record the selected-index digest and date coverage and include:

- uniform legal-action and training-frequency baselines;
- end-to-end and move-versus-switch accuracy;
- explicitly labeled oracle-type accuracy;
- top-1/top-2/top-3 accuracy and mean reciprocal rank;
- candidate-set NLL, entropy, and top-action margin;
- ordinary-move, switch, and Terastallized-move slices.

Train the compact non-language-model baseline on the same chronological data:

```bash
pokemon-baseline static \
  --data-file data/gen9ou-dev/validation.jsonl \
  --baseline-train-file data/gen9ou-dev/train.jsonl \
  --max-examples 5000 \
  --sample-mode hash \
  --sample-seed 42 \
  --output reports/static-baselines-validation.json

pokemon-baseline train \
  --train-file data/gen9ou-dev/train.jsonl \
  --validation-file data/gen9ou-dev/validation.jsonl \
  --output-dir outputs/hashed-action-baseline \
  --max-train-examples 100000 \
  --max-validation-examples 5000 \
  --sample-mode hash

pokemon-baseline evaluate \
  --checkpoint outputs/hashed-action-baseline/best \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 5000 \
  --sample-mode hash \
  --sample-seed 42 \
  --baseline-train-file data/gen9ou-dev/train.jsonl \
  --output reports/hashed-action-baseline-validation.json
```

This model embeds deterministic hashes of structured state and candidate-action
features and scores their interaction with a small network. It is not a language
model, uses the same legal mask, and tests whether the 0.5B causal model earns its
additional cost.

There is also a stricter mechanics-only ablation. It receives the exact same
numeric and categorical candidate inputs as the hybrid model and no Qwen
representation:

```bash
pokemon-mechanics-baseline train \
  --train-file data/gen9ou-dev/train.jsonl \
  --validation-file data/gen9ou-dev/validation.jsonl \
  --output-dir outputs/mechanics-only-baseline

pokemon-mechanics-baseline evaluate \
  --checkpoint outputs/mechanics-only-baseline/best \
  --data-file data/gen9ou-dev/validation.jsonl \
  --output reports/mechanics-only-validation.json
```

If the mechanics-only network matches the hybrid model, Qwen is not adding
enough state value to justify its compute. If the hybrid wins clearly, the gap
measures what the learned state representation contributes beyond the engineered
candidate mechanics.

### Selective logits

Masking labels with `-100` prevents prompt tokens from contributing to causal-LM
loss, but a normal causal-LM forward pass can still project every prompt hidden
state through Qwen's 151,936-column vocabulary head. Selective logits ask Qwen to
perform that projection only at the three or four positions that predict the
action label and EOS. The transformer still processes the complete prompt; this
removes unnecessary vocabulary projection, not attention or MLP work.

The default `--loss-projection auto` uses selective projection when supported.
`--loss-projection selective` requires it. For a controlled SFT benchmark,
compare it with `--loss-projection full` in separate processes using identical
examples and sequence lengths. Compare synchronized examples/second and peak
allocated VRAM, not only `nvidia-smi` or reserved allocator memory. The policy
head does not need this comparison because it replaces the vocabulary head with
13 outputs.

## 1. Install and verify the environment

Run all commands from the repository root:

```bash
cd /home/dylan/Gitrepos/allrepos/pokemonBattler
uv pip install -e .
python -m unittest discover -v
```

Check the GPU and compute dtype:

```bash
python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); print("BF16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)'
```

`--dtype auto` uses BF16 when the last line is `BF16: True`; otherwise it uses
FP16. Do not force BF16 on unsupported hardware.

The training commands use `--local-files-only`. Confirm the base checkpoint is
already cached:

```bash
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True); print("cached")'
```

If this fails, download the official checkpoint once while online:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B")'
```

The local replay archive used below is:

```text
data/raw/metamon/gen9ou.tar.gz
```

Do not extract it. `pokemon-prepare` streams it directly.

## 2. Create the chronological development dataset

Use one fixed chronological definition throughout the project:

```text
train       before 2026-01-01
validation  2026-01-01 through 2026-03-31
test        2026-04-01 and later
```

Both perspectives and every sampled turn from one battle remain in the same
split. The test split must not be evaluated until the model configuration has
been frozen.

Create a 2% turn sample across the entire archive:

```bash
pokemon-prepare \
  --input data/raw/metamon/gen9ou.tar.gz \
  --output-dir data/gen9ou-dev \
  --format gen9ou \
  --min-rating 1600 \
  --outcome both \
  --split-mode chronological \
  --validation-start 2026-01-01 \
  --test-start 2026-04-01 \
  --sample-rate 0.02 \
  --seed 42
```

This operation must scan the compressed archive and can take a while. The
sampling is deterministic, so the same seed and rate reproduce the same turns.
Turn-level sampling is intentional: it spreads a limited training budget over
many battles instead of retaining long runs of highly correlated adjacent turns.

Do not add `--max-examples-per-split` here. The archive is date ordered, so a
hard cap would preferentially retain the earliest examples in each time window.

Inspect the preparation report:

```bash
python -m json.tool data/gen9ou-dev/prepare_report.json
wc -l data/gen9ou-dev/train.jsonl \
      data/gen9ou-dev/validation.jsonl \
      data/gen9ou-dev/test.jsonl
```

The development set is large enough when it contains at least:

```text
train       50,000 rows
validation   5,000 rows
test         5,000 rows
```

If any split is smaller, recreate it at 4%:

```bash
pokemon-prepare \
  --input data/raw/metamon/gen9ou.tar.gz \
  --output-dir data/gen9ou-dev \
  --format gen9ou \
  --min-rating 1600 \
  --outcome both \
  --split-mode chronological \
  --validation-start 2026-01-01 \
  --test-start 2026-04-01 \
  --sample-rate 0.04 \
  --seed 42 \
  --overwrite
```

## 3. Measure prompt lengths before training

The collator pads only to the longest sequence in the current microbatch.
`--max-length 4096` is a safety ceiling, not a request to pad every row to 4096.

Measure 25,000 training prompts:

```bash
mkdir -p reports
pokemon-inspect \
  --data-file data/gen9ou-dev/train.jsonl \
  --model Qwen/Qwen2.5-0.5B \
  --local-files-only \
  --max-examples 25000 \
  --max-length 4096 \
  --example-output reports/example-rendered-prompt.md \
  --output reports/gen9ou-dev-train-stats.json
python -m json.tool reports/gen9ou-dev-train-stats.json
```

Repeat on validation:

```bash
pokemon-inspect \
  --data-file data/gen9ou-dev/validation.jsonl \
  --model Qwen/Qwen2.5-0.5B \
  --local-files-only \
  --max-examples 5000 \
  --max-length 4096 \
  --output reports/gen9ou-dev-validation-stats.json
```

Proceed when `prompts_over_max_length` is zero in both reports. Do not silently
use `--truncation left`: that can remove the task header or early battle state.
If real prompts exceed 4096, first make the serializer more compact and rerun
the inspection. Increasing context beyond 4096 is the last option on an 8 GB
GPU.

`mechanics-v2` keeps only compact move-name lists and history in text; all
numeric move prose still travels through the structured branch. On the first
100 training rows, its median was 592 tokens versus 463 for v1 and 1,207 for
`compact-v1`; its maximum was 707 versus 1,415 for `compact-v1`. Run
`pokemon-inspect --prompt-format mechanics-v2` before changing the context
ceiling. The v1 serializer remains available for its original checkpoints.

## Legacy causal-LM experiment protocol

The remaining numbered sections document the original generative-SFT research
protocol. They are retained for reproducibility, but they are not the current
recommended run and some of their budgets intentionally describe the old
objective. Use the one-command mechanics-head experiment at the top of this
README for the next model run. When reproducing a command in this legacy
section, add `--objective sft --prompt-format verbose-v1` explicitly.

## 4. Record the unfine-tuned validation baseline

This establishes whether SFT adds value over the pretrained model:

```bash
pokemon-evaluate \
  --model Qwen/Qwen2.5-0.5B \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 1000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/base-validation.json
```

This is validation data, not test data.

## 5. Pass the 128-example memorization gate

This checkpoint is disposable. Dropout is disabled and the learning rate is
deliberately high because the only question is whether the pipeline can
memorize a tiny slice.

```bash
pokemon-train \
  --model Qwen/Qwen2.5-0.5B \
  --train-file data/gen9ou-dev/train.jsonl \
  --output-dir outputs/overfit-128 \
  --method lora \
  --local-files-only \
  --load-in-4bit \
  --dtype auto \
  --gradient-checkpointing \
  --overfit-examples 128 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-3 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0 \
  --epochs 100 \
  --max-steps 400 \
  --max-length 4096 \
  --eval-steps 0 \
  --log-steps 10
```

At startup, the trainer should print:

```json
{"loss_projection": "supervised_positions_only", "logits_parameter": "logits_to_keep"}
```

This confirms that Qwen computes vocabulary logits only for the action and EOS
positions. Each training log also reports current, reserved, and peak allocated
VRAM. Computing logits for every masked prompt position is mathematically
unnecessary here and can exhaust an 8 GB GPU because Qwen's vocabulary contains
151,936 tokens.

Evaluate exactly those rows:

```bash
pokemon-evaluate \
  --adapter outputs/overfit-128/final \
  --data-file data/gen9ou-dev/train.jsonl \
  --max-examples 128 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/overfit-128.json
```

Do not continue unless same-slice accuracy is at least 95% and the loss became
very small. Failure means the harness, labels, masking, or optimizer needs
debugging. Success proves only that the pipeline is connected; it says nothing
about generalization.

## 6. Select hyperparameters on validation only

The fixed search budget is 2,000 optimizer updates per run. A microbatch of one
is the safest choice for 8 GB because Qwen has a large output vocabulary and
the battle prompts have variable lengths. Accumulating 32 microbatches preserves
an effective batch of 32 examples.

### 6.1 Learning-rate screen

Run the same rank-16 configuration at three learning rates:

```bash
for PB_LR in 1e-4 2e-4 4e-4
do
  pokemon-train \
    --model Qwen/Qwen2.5-0.5B \
    --train-file data/gen9ou-dev/train.jsonl \
    --validation-file data/gen9ou-dev/validation.jsonl \
    --output-dir "outputs/dev-lr-${PB_LR}" \
    --method lora \
    --local-files-only \
    --load-in-4bit \
    --dtype auto \
    --gradient-checkpointing \
    --batch-size 1 \
    --eval-batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate "${PB_LR}" \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --epochs 100 \
    --max-steps 2000 \
    --max-length 4096 \
    --validation-examples 1024 \
    --eval-batches 512 \
    --eval-steps 250 \
    --log-steps 20 \
    --seed 42
done
```

For the legacy SFT objective, each run saves the lowest-validation-loss adapter
under `best/`.
Evaluate those three adapters on the same 2,000 validation rows:

```bash
for PB_LR in 1e-4 2e-4 4e-4
do
  pokemon-evaluate \
    --adapter "outputs/dev-lr-${PB_LR}/best" \
    --data-file data/gen9ou-dev/validation.jsonl \
    --max-examples 2000 \
    --max-length 4096 \
    --local-files-only \
    --load-in-4bit \
    --output "reports/dev-lr-${PB_LR}-validation.json"
done
```

Select the learning rate with the highest validation action accuracy. If two
runs are within 0.5 percentage points, use validation loss as the tiebreaker. If
they are still effectively tied, keep `2e-4`.

### 6.2 Rank screen

Keep the winning learning rate fixed. The rank-16 run already exists, so train
only ranks 8 and 32:

```bash
export PB_BEST_LR=2e-4

for PB_RANK in 8 32
do
  PB_ALPHA=$((2 * PB_RANK))
  pokemon-train \
    --model Qwen/Qwen2.5-0.5B \
    --train-file data/gen9ou-dev/train.jsonl \
    --validation-file data/gen9ou-dev/validation.jsonl \
    --output-dir "outputs/dev-rank-${PB_RANK}" \
    --method lora \
    --local-files-only \
    --load-in-4bit \
    --dtype auto \
    --gradient-checkpointing \
    --batch-size 1 \
    --eval-batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate "${PB_BEST_LR}" \
    --lora-rank "${PB_RANK}" \
    --lora-alpha "${PB_ALPHA}" \
    --lora-dropout 0.05 \
    --epochs 100 \
    --max-steps 2000 \
    --max-length 4096 \
    --validation-examples 1024 \
    --eval-batches 512 \
    --eval-steps 250 \
    --log-steps 20 \
    --seed 42
done
```

Replace `PB_BEST_LR=2e-4` with the measured winner before running this command.
Evaluate all three ranks on the same validation rows:

```bash
pokemon-evaluate \
  --adapter outputs/dev-rank-8/best \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 2000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/dev-rank-8-validation.json

pokemon-evaluate \
  --adapter "outputs/dev-lr-${PB_BEST_LR}/best" \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 2000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/dev-rank-16-validation.json

pokemon-evaluate \
  --adapter outputs/dev-rank-32/best \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 2000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/dev-rank-32-validation.json
```

Choose the highest validation action accuracy. Within 0.5 percentage points,
choose lower validation loss; if still tied, choose the lower rank.

### 6.3 Dropout check

Train one final development run with the winning learning rate and rank but no
LoRA dropout. Set these values to the measured winners:

```bash
export PB_BEST_LR=2e-4
export PB_BEST_RANK=16
export PB_BEST_ALPHA=32

pokemon-train \
  --model Qwen/Qwen2.5-0.5B \
  --train-file data/gen9ou-dev/train.jsonl \
  --validation-file data/gen9ou-dev/validation.jsonl \
  --output-dir outputs/dev-dropout-0 \
  --method lora \
  --local-files-only \
  --load-in-4bit \
  --dtype auto \
  --gradient-checkpointing \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate "${PB_BEST_LR}" \
  --lora-rank "${PB_BEST_RANK}" \
  --lora-alpha "${PB_BEST_ALPHA}" \
  --lora-dropout 0 \
  --epochs 100 \
  --max-steps 2000 \
  --max-length 4096 \
  --validation-examples 1024 \
  --eval-batches 512 \
  --eval-steps 250 \
  --log-steps 20 \
  --seed 42
```

Evaluate it on the same validation rows:

```bash
pokemon-evaluate \
  --adapter outputs/dev-dropout-0/best \
  --data-file data/gen9ou-dev/validation.jsonl \
  --max-examples 2000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/dev-dropout-0-validation.json
```

Compare that report with the matching 0.05-dropout rank report using the same
decision rule. At this point, freeze:

```text
learning rate
LoRA rank
LoRA alpha = 2 × rank
LoRA dropout
```

Do not search on the test split.

## 7. Create the main training dataset

After freezing hyperparameters, increase the deterministic sample from 2% to
10%. This is a compute-conscious first main run, not a claim that 10% is
universally optimal.

```bash
pokemon-prepare \
  --input data/raw/metamon/gen9ou.tar.gz \
  --output-dir data/gen9ou-main \
  --format gen9ou \
  --min-rating 1600 \
  --outcome both \
  --split-mode chronological \
  --validation-start 2026-01-01 \
  --test-start 2026-04-01 \
  --sample-rate 0.10 \
  --seed 42
```

Inspect the report, row counts, and prompt lengths again:

```bash
python -m json.tool data/gen9ou-main/prepare_report.json
wc -l data/gen9ou-main/train.jsonl \
      data/gen9ou-main/validation.jsonl \
      data/gen9ou-main/test.jsonl
pokemon-inspect \
  --data-file data/gen9ou-main/train.jsonl \
  --model Qwen/Qwen2.5-0.5B \
  --local-files-only \
  --max-examples 50000 \
  --max-length 4096 \
  --output reports/gen9ou-main-train-stats.json
```

## 8. Run the main SFT

The first main budget is the smaller of:

```text
three passes over the sampled training rows
20,000 optimizer updates
```

Calculate it for an effective batch of 32:

```bash
PB_TRAIN_ROWS=$(wc -l < data/gen9ou-main/train.jsonl)
PB_UPDATES_PER_EPOCH=$(( (PB_TRAIN_ROWS + 31) / 32 ))
PB_MAIN_STEPS=$(( 3 * PB_UPDATES_PER_EPOCH ))
if [ "${PB_MAIN_STEPS}" -gt 20000 ]; then PB_MAIN_STEPS=20000; fi
printf 'rows=%s updates_per_epoch=%s main_steps=%s\n' \
  "${PB_TRAIN_ROWS}" "${PB_UPDATES_PER_EPOCH}" "${PB_MAIN_STEPS}"
```

Set the frozen hyperparameters and train:

```bash
export PB_BEST_LR=2e-4
export PB_BEST_RANK=16
export PB_BEST_ALPHA=32
export PB_BEST_DROPOUT=0.05

pokemon-train \
  --model Qwen/Qwen2.5-0.5B \
  --train-file data/gen9ou-main/train.jsonl \
  --validation-file data/gen9ou-main/validation.jsonl \
  --output-dir outputs/qwen25-05b-main \
  --method lora \
  --local-files-only \
  --load-in-4bit \
  --dtype auto \
  --gradient-checkpointing \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate "${PB_BEST_LR}" \
  --lora-rank "${PB_BEST_RANK}" \
  --lora-alpha "${PB_BEST_ALPHA}" \
  --lora-dropout "${PB_BEST_DROPOUT}" \
  --epochs 3 \
  --max-steps "${PB_MAIN_STEPS}" \
  --max-length 4096 \
  --validation-examples 2048 \
  --eval-batches 1024 \
  --eval-steps 500 \
  --log-steps 20 \
  --seed 42
```

Replace the four exported defaults with the measured development winners. The
cosine schedule spans the calculated main budget. The trainer saves:

```text
outputs/qwen25-05b-main/best/   selected validation checkpoint
outputs/qwen25-05b-main/final/  parameters at the last update
outputs/qwen25-05b-main/history.json
```

Compare `best/` and `final/` on the same validation rows:

```bash
for PB_CHECKPOINT in best final
do
  pokemon-evaluate \
    --adapter "outputs/qwen25-05b-main/${PB_CHECKPOINT}" \
    --data-file data/gen9ou-main/validation.jsonl \
    --max-examples 5000 \
    --max-length 4096 \
    --local-files-only \
    --load-in-4bit \
    --output "reports/qwen25-05b-main-${PB_CHECKPOINT}-validation.json"
done
```

Use `best/` unless `final/` has higher full validation action accuracy. Freeze
that choice before opening the test results:

```bash
export PB_SELECTED_CHECKPOINT=best
```

The training curve determines the next experiment:

- If the best validation result occurs well before the end and later validation
  loss rises, do not add more updates.
- If validation loss and action accuracy are still improving at the end, repeat
  the main run with a 40,000-update cap.
- If training loss falls while validation metrics do not improve, more epochs
  will not solve the generalization problem.
- If both losses remain high, revisit learning rate, prompt representation, data
  quality, or model capacity.

## 9. Evaluate the frozen model on test once

First evaluate the unfine-tuned base on exactly 5,000 test rows:

```bash
pokemon-evaluate \
  --model Qwen/Qwen2.5-0.5B \
  --data-file data/gen9ou-main/test.jsonl \
  --max-examples 5000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/base-test.json
```

Then evaluate the selected adapter on the same rows:

```bash
pokemon-evaluate \
  --adapter "outputs/qwen25-05b-main/${PB_SELECTED_CHECKPOINT}" \
  --data-file data/gen9ou-main/test.jsonl \
  --max-examples 5000 \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit \
  --output reports/qwen25-05b-main-test.json
```

Report at minimum:

- base and SFT action accuracy;
- move versus switch accuracy;
- target and prediction action distributions;
- average top-1 score margin;
- dataset filters, date boundaries, sample rate, and seed;
- the complete training configuration and selected checkpoint step.

Accuracy is agreement with the recorded action. It is not battle win rate.

## 10. Predict one recorded state

`pokemon-predict` accepts either a raw Metamon state object or one prepared JSONL
row saved as JSON:

```bash
pokemon-predict \
  --adapter "outputs/qwen25-05b-main/${PB_SELECTED_CHECKPOINT}" \
  --state-file /path/to/state.json \
  --max-length 4096 \
  --local-files-only \
  --load-in-4bit
```

It scores only the listed legal actions and returns both the action ID and its
semantic move or switch mapping.

## 11. What remains before claiming a strong battler

The current repository measures offline imitation but does not yet run complete
Showdown battles. The next implementation milestone is a local simulator adapter
that uses the live server's legal-action mask.

Evaluate each frozen policy against the same:

1. team pool;
2. random seeds;
3. number of battles;
4. opponents: Random, fixed heuristics, and frozen Metamon policies.

Use at least several hundred battles per matchup and report win-rate confidence
intervals. This metric, not offline action accuracy, should select a policy for
later preference optimization or simulator-based RL.

Metamon documents a replay-to-live simulator gap: spectator replays omit some
information sent to live players. Treat replay SFT as offline pretraining and
expect self-collected simulator data to be necessary for the strongest policy.

## Why these defaults

- **NF4 and double quantization:** the QLoRA configuration designed for training
  a frozen 4-bit base model while updating LoRA parameters.
- **BF16/FP16 auto selection:** uses the most stable supported 16-bit compute
  dtype without assuming the GPU supports BF16.
- **All transformer projections:** QLoRA-style adaptation covers Qwen's
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`.
- **Rank 16, alpha 32:** a moderate starting capacity with scaling
  `alpha / rank = 2`; ranks 8 and 32 are explicitly tested.
- **Learning rates 1e-4, 2e-4, and 4e-4:** a small log-scale search around a
  conventional QLoRA starting point.
- **Dropout 0 and 0.05:** directly tests whether regularization helps this data
  volume.
- **Effective batch 32:** stable enough for a first action-learning experiment
  while microbatch one protects an 8 GB GPU.
- **Cosine decay, 3% warmup, clipping at 1.0:** conservative optimizer defaults
  held fixed while the parameters most likely to matter are searched.

These values are not declared optimal in advance. The protocol defines
"selected" as best on held-out chronological validation data under an equal
training budget. That is the strongest defensible conclusion before simulator
evaluation.

## Action semantics

Metamon uses 13 universal actions:

- `A0`–`A3`: active moves in alphabetical order
- `A4`–`A8`: currently available, non-fainted switches in alphabetical order
- `A9`–`A12`: the same four moves with terastallization

The identity attached to a switch action can change after a knockout. If one
switch candidate faints, later candidates shift down. Every prompt therefore
includes mappings such as `A4 -> switch Corviknight`.

The replay mask matches Metamon's `maybe_valid_actions`. Live play must use the
simulator's stricter current mask so trapping, disabled moves, choice locks, and
other mechanics are enforced.

## References and data license

- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT quantization guide](https://huggingface.co/docs/peft/main/developer_guides/quantization)
- [Metamon repository and environment documentation](https://github.com/UT-Austin-RPL/metamon)
- [Metamon parsed replay dataset](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)

The current Hugging Face dataset is marked `CC-BY-NC-4.0`. Review its current
license and attribution requirements before publishing checkpoints, derived
data, or a hosted demo.
