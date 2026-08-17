# Large offline self-play without four copies of Qwen

The earlier training rounds kept changing the loss while leaving the data
coverage almost fixed. The Foul Play collections contain roughly 51,000 turn
labels from 1,000 battles, but only six training opponent teams. They include
soft policies and action values, so collecting the same setup again is not a
new learning signal. It is more repetition of the same matchups.

This pipeline changes the source of data and the cost of using it. It imports
the official Metamon self-play corpora instead of trying to manufacture a large
dataset from public ladder games:

- `pac-base`: 11 million trajectories across its supported formats;
- `pac-exploratory`: 7 million trajectories collected at a higher sampling
  temperature;
- `gl_05_26`: 139,000 Gen 9 general-ladder team files;
- `hl_05_26`: 43,000 Gen 9 high-ladder team files.

Those counts are for the published corpora, not the exact number of Gen 9 OU
rows selected by this project. The run reports the actual source trajectories,
prepared decisions, split sizes, invalid rows, and action-mapping failures.
Metamon's current data and model descriptions are in its
[official repository](https://github.com/UT-Austin-RPL/metamon).

## One command

Run this from the repository root in the already-activated environment:

```bash
python -m pokemon_battler.large_offline_pipeline \
  --output-dir outputs/metamon-large-v1
```

The first run downloads large archives, but it does not expand them into
uncompressed tar files. Each outer `.tar.lz4` is read as one sequential stream.
The member filename is checked against the deterministic trajectory sample
before its inner JSON.LZ4 payload is decoded or submitted to a worker. The
prepared rows still retain the visible roster and history, so their exact disk
cost depends on the selected battles. The command prints free disk at startup
but does not guess a safe cutoff.

The two downloaded Gen 9 OU archives contain 6,961,526 trajectories. The
default selects a deterministic 0.5%, projected from the measured 5% partial
run to retain about 34,800 complete trajectories and 1.27 million decisions.
That is a coverage/space compromise, not a claim that 0.5% is optimal. The
default storage guards stop prepared output above 32 GiB and stop before cache
construction if its exact shape-based estimate exceeds 16 GiB. These can be
changed with `--maximum-prepared-gib` and `--maximum-cache-gib`.

To use every Gen 9 trajectory on a machine with hundreds of GiB available:

```bash
python -m pokemon_battler.large_offline_pipeline \
  --output-dir outputs/metamon-large-full-v1 \
  --trajectory-sample-rate 1 \
  --maximum-prepared-gib 300 \
  --maximum-cache-gib 120
```

To use data that has already been downloaded under `data/metamon-large/`, add
`--skip-download`. A completed output directory is returned immediately. An
interrupted preparation resumes after its last committed shard; completed
feature-cache shards are also reused. Because compressed TAR.LZ4 is sequential,
resume rescans the source stream up to that shard without decoding sampled-out
members. It does not recreate an uncompressed archive. By default, downloaded
self-play archives are deleted only after the entire pipeline completes. Add
`--keep-compressed` if they should remain available for a separate future run.

The source contains a small fraction of recorded actions that cannot be
reconstructed from the visible legal-action state. Those rows are dropped and
reported. The large pipeline permits at most 1%; the measured partial corpus
was approximately 0.44%. This is still a compatibility gate, but it no longer
rejects a corpus with more than 99% recoverable action parity.

## Continue the promoted sidecar

The completed v1 run retained 34,524 trajectories and 1,249,105 decisions. Its
candidate won 48 of 100 paired held-out games; the unchanged source champion
won 30 of 100 on the identical opponent-team schedule. The paired win-rate
delta was `+0.18`, with a bootstrap interval of `[+0.07, +0.29]`.

Continuation must load `structured_policy_head.safetensors`, not initialize a
new sidecar from `interaction_head.safetensors`. It must also avoid replaying
the same deterministic sample as though it were new data. The v2 command does
both:

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

With seed 42, v1 selected hash values in `[0, 0.005)` and v2 selects
`[0.005, 0.010)`. These intervals are mutually exclusive by construction. The
rehearsal ratio is a fraction of the combined training set: enough old cached
rows are sampled for 25% old and 75% new examples. Validation uses only the new
slice. Epoch zero is evaluated and participates in checkpoint selection, so
the saved sidecar can roll back to the exact input weights.

The initial `0.5` deployment blend was not tuned. V2 runs the four requested
weights against one shared champion schedule built from validation-team
compositions, writes the selected weight into the candidate checkpoint, and
then evaluates candidate and champion on a separate test-team schedule. A
strict `--minimum-delta-interval-lower 0` requires the paired interval's lower
bound to be positive before promotion.

## Completed v2 result

I ran the continuation on the disjoint `[0.005, 0.010)` hash window. It retained
34,735 trajectories and 1,259,031 transitions, mixed 377,170 v1 rehearsal
examples into each epoch, and selected a sidecar blend weight of 0.75. On the
separate 200-game held-out schedule, v2 finished 109-91 while v1 finished
96-104.

The automatic pointer remained on v1 because the paired interval for the
difference crossed zero. I kept that strict gate intact and used v2 as the next
public candidate because it had the stronger point estimate in both battle
stages. The frozen public run finished 53-47 over 100 Generation 9 OU ladder
games with zero fallbacks across 2,852 decisions. The strongest measured
checkpoint is `outputs/metamon-large-v2/04-candidate`.

## Where four-way parallelism is useful

The default `--workers 4` is used in four CPU-bound places:

1. trajectory JSON and inner LZ4 decoding;
2. construction of mechanics, roster, history, and candidate features;
3. syntax parsing, composition deduplication, and splitting of team files;
4. data loading and prefetch during sidecar training.

The held-out local suite separately uses `--concurrent-games 4`. These are four
Showdown/Foul Play games sharing one loaded policy runtime, not four independent
training jobs.

Launching four Qwen training or encoding processes on the same GPU would not
give the same benefit. Each process would load another model and fight for VRAM
and memory bandwidth. This pipeline therefore does not encode every offline row
with Qwen. It trains the structured part in large batches and loads Qwen once
for deployment and battle evaluation.

Uncompressed `.tar`, legacy `.tar.gz`, and official `.tar.lz4` inputs are all
streamed directly. Compressed archives have one sequential outer decompression
stream, while selected inner JSON parsing, LZ4 decoding, state enrichment, and
feature generation are pipelined across workers. The worker queue is bounded so
a large archive cannot fill RAM with pending battle payloads.

## The deployed model

The output is still a Qwen checkpoint. It contains:

- the accepted batch-005 Qwen LoRA adapter;
- the accepted interaction policy head;
- the existing team-preview head when the source checkpoint provides one;
- `structured_policy_head.safetensors`, trained on the large self-play sample.

The sidecar uses the exact numeric and categorical tensors already supplied to
the interaction Transformer: field conditions, roster state, revealed
identities, status/effects, legal move and switch mechanics, candidate actor,
and four recent transition events. It does not use Pokémon names as free-form
prompt prose. Species, moves, items, abilities, types, and effects remain
categorical identities because collapsing Stealth Rock, Recover, Encore, and
every other non-damage move into one flag destroyed strategically important
distinctions.

Training does not discard losing trajectories or imitate every exploratory
action with equal weight. The selected action's final WIN/LOSS result trains a
separate action-value head; a state-value head uses expectile regression; and
the actor applies advantage-weighted behavior cloning with a small ordinary
cloning term. Losing games therefore supply negative value targets while still
contributing their locally useful actions. This is logged-action offline RL,
not a counterfactual label for every legal move.

At inference, the normal Qwen interaction head produces a legal 13-action log
distribution. The structured sidecar produces another legal distribution. The
runtime adds half of the sidecar log probability by default and renormalizes:

```text
logits = qwen_log_probability + 0.5 * structured_log_probability
```

`--blend-weight` changes that coefficient for a new training run. A zero value
is exactly the old Qwen policy. The source checkpoint is copied, never edited.

This is deliberately different from the failed trajectory and residual runs.
The trajectory run trained a new actor on 288,081 cached replay rows and
replaced the champion distribution. The residual run trained on 10,000 selected
teacher states from the small six-team collection. The new sidecar keeps the
champion distribution and changes the dataset by orders of magnitude.

## Action parity and split leakage

Metamon and this repository both use 13 universal action slots:

```text
A0-A3   alphabetically ordered moves
A4-A8   alphabetically ordered available switches
A9-A12  the same move slots with Terastallization
```

Preparation reconstructs the legal action set for every retained decision and
checks that the recorded action maps into it. `01-prepared/manifest.json`
reports `checked_actions`, `unmapped_actions`, and `unmapped_fraction`. It does
not silently remap an unknown action to A0.

Train, validation, and test assignment hashes the battle ID, so the two player
points of view from one battle cannot cross splits. Team manifests are split by
an order-independent six-species composition hash, not filename. Duplicate
exports and duplicate compositions are removed before the split.

## Artifacts

```text
outputs/metamon-large-v1/
  01-prepared/
    manifest.json
    train/part-*.jsonl
    validation/part-*.jsonl
    test/part-*.jsonl
  02-interaction-cache/
    manifest.json
    train/part-*.interaction-v1/
    validation/part-*.interaction-v1/
    test/part-*.interaction-v1/
  03-team-manifests/
    train.txt
    validation.txt
    test.txt
    report.json
  04-candidate/
    structured_policy_head.safetensors
    structured_training_report.json
    training_config.json
    ...copied Qwen checkpoint files...
  05-blend-sweep/
    champion/
    candidate-blend-*/
    summary.json
  05-heldout-evaluation/
    candidate/
    champion/
    summary.json
  selected_checkpoint.txt
  run_manifest.json
```

Prepared JSONL files get reusable binary byte-offset indexes. Feature caches are
NumPy memory maps. Actions, outcomes, and battle lengths are cached separately,
so every training epoch does not parse the large JSON rows again.

## What this can and cannot fix

This can fix the most obvious scale problem: too few teams, too few visited
states, and too little switching/Tera/setup variety. Higher-temperature
self-play also contains more suboptimal actions around which a value model can
learn, instead of only repeating one deterministic teacher line.

It cannot guarantee a 51% public win rate. Logged actions still do not reveal
the counterfactual next state for every alternative. A large self-play policy
can carry its own biases. The current sidecar has four events of explicit
history, not an unlimited belief state, and the final blend coefficient is a
hyperparameter rather than a proof of strength. The local Foul Play gate is a
regression check, not a substitute for a new public ladder measurement.

The important difference is that a failure is now informative. If the sidecar
sees hundreds of thousands of diverse trajectories, improves held-out action
ranking, and still cannot improve paired battle results, the remaining problem
is no longer “collect the same 200 teacher games again.” It points at teacher
quality, hidden-state belief, long-horizon credit, or the 0.5B Qwen contribution
itself.
