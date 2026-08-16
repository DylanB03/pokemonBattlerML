# Conservative champion residual

## What this is trying to fix

The last architecture experiment did not improve full-game results. The
recurrent policy scored 16 wins in 100 games against the memoryless policy's 22.
The memoryless policy then scored 20 against the existing batch-005 champion's
21. Offline action accuracy moved slightly, but the battle result did not.

The important implementation error was not a learning-rate choice. The new
trajectory actor replaced the champion's learned action distribution with a
randomly initialized policy head. It had to relearn the decision boundary before
its memory or value targets could help. That is too much risk for evidence that
was already weak.

This pipeline treats the current champion as the fallback, not as disposable
initialization.

## One command

Run this from the repository root in the already-active environment:

```bash
python -m pokemon_battler.residual_pipeline \
  --output-dir outputs/champion-residual-v1
```

Use a new output directory for another attempt. The runner refuses to merge a
new experiment into an existing directory.

## The policy that is trained

The deployed champion already produces a normalized legal-action distribution
`log p_champion(a | s)`. The new head sees the champion's frozen global and 13
candidate embeddings and predicts one bounded correction per candidate:

```text
delta(a, s) = 1.5 * tanh(MLP(global, candidate, global * candidate, champion log p))
log p_new(a | s) = log_softmax(log p_champion(a | s) + delta(a, s))
```

The final residual projection is initialized to all zeros. On 32 real rows from
the existing validation cache, the untrained residual changed zero top actions;
the largest legal log-probability difference was `9.54e-7`, from float32
renormalization. It therefore begins as the same operational policy rather than
a new random actor.

The correction is capped at 1.5 logits. This is large enough to reverse a close
decision but prevents a small teacher set from producing unbounded confidence.
Qwen, its LoRA adapter, the interaction transformer, its hierarchical policy
scorers, and the value heads remain frozen.

## Data and leakage controls

The defaults reuse the two completed smart-teacher traces:

```text
outputs/qwen-dagger-v1/round-00/teacher/foul_play_teacher.jsonl
outputs/qwen-dagger-v1/round-01/teacher/foul_play_teacher.jsonl
```

Those traces contain 50,996 turn decisions from 1,000 local games. The player's
team is the same fixed balance team used by this project. The opponent cycles
among six validated teams. Foul Play supplies a complete 13-slot search
distribution, not only its argmax.

The selector hashes whole battle IDs into train and validation partitions before
choosing rows. With the defaults, it selects:

- 8,000 training decisions from 503 battles;
- 2,000 validation decisions from 204 different battles;
- no battle overlap between those partitions;
- a 60% attack, 35% switch, and 5% Tera target mix;
- no more than 24 decisions from one battle;
- balanced coverage across the six teacher enemy teams.

The 35% switch allocation is deliberate. Public runs showed attack repetition
and almost no voluntary switching. Sampling the raw action frequency again
would preserve that failure. The targets are still the teacher's soft
distribution, so this does not turn every selected state into a forced switch.

Three enemy teams in the nine-team manifest—`gen9ou6`, `gen9ou7`, and
`gen9ou8`—never appear in teacher training. They are reserved for the first
complete-game gate.

## Reusing the expensive cache

The pipeline reuses:

```text
outputs/trajectory-iql-v1/02-encoded-cache/train       288,081 rows
outputs/trajectory-iql-v1/02-encoded-cache/validation  29,021 rows
```

Both caches have the exact signature of
`outputs/public-learning/positive-winrate-1000/batch-005/candidate`. Training
samples 32,000 broad replay rows and validation samples 8,000. The preservation
loss is `KL(champion || residual)` on those states. This constrains the residual
away from teacher-selected situations instead of hoping that 8,000 decisions
represent all of OU.

Only the 10,000 teacher rows need a new Qwen and interaction-encoder pass. They
are written as memory-mapped global and candidate embeddings. Residual training
after that is a small PyTorch-head job and does not load Qwen.

Before creating the run directory, the command verifies that both caches use
the selected checkpoint's exact signature and embedding size. It also refuses
a source checkpoint that already contains a trajectory or residual actor. That
restriction is deliberate: silently stacking a new correction over an
auxiliary policy would make the cached starting distribution differ from the
one used in live play.

## Training and offline gate

The objective has three terms:

```text
soft teacher cross-entropy
+ 0.5 * broad-replay KL from the champion
+ 0.01 * squared residual-logit penalty
```

Teacher rows receive a modest confidence weight. Validation is run after every
epoch, the best replay-preserving epoch is restored, and training stops after
three stale epochs. A lower teacher loss can never displace an earlier eligible
checkpoint if it violates either replay-drift limit.
The candidate can proceed only if:

- teacher KL improves by at least 0.02, or teacher top-1 agreement improves by
  at least 2 percentage points;
- replay KL from the champion is at most 0.05;
- at most 15% of replay top actions change.

These are screening rules, not proof of a stronger player. Passing them only
earns the right to run battles.

## Battle gates and promotion

The first gate runs 50 paired games for both candidate and champion against the
same Foul Play team schedule, using only the three unseen enemy teams. If the
candidate does not produce a positive delta with the repository's paired
bootstrap guard, the run stops.

If it passes, a second gate runs 500 games per policy over all nine enemy teams.
Its observed win-rate delta must be positive and the lower end of the paired
95% bootstrap interval must be above zero. A positive result whose interval
still includes zero is rejected as inconclusive. The larger final budget is
spent only after the cheap gates pass; 100 games was too underpowered for this
strict promotion rule. The source champion remains selected in every other
case.

This is intentionally conservative because a 51% target cannot be established
from training loss, teacher agreement, or one noisy win-rate sample. A positive
local paired result is still not a public-ladder guarantee. It is the minimum
evidence required before replacing the best checkpoint currently available.

## Artifacts

```text
outputs/champion-residual-v1/
  manifest.json
  selected_checkpoint.txt
  01-selected-teacher/
    teacher-train.jsonl
    teacher-validation.jsonl
  02-teacher-cache/
    train/
    validation/
  03-residual-candidate/
    residual_head.safetensors
    residual_config.json
    residual_training_report.json
    ...copied champion checkpoint files
  04-heldout-pilot/
  05-full-gate/
```

`selected_checkpoint.txt` is written before data preparation. It initially
contains the source champion's absolute path, remains there if an exception
occurs, and changes
to `03-residual-candidate` only after the final battle gate passes. The candidate
and all rejection evidence remain available either way.

`manifest.json` records each completed phase. The residual training report
contains before/after teacher metrics, broad-replay drift, every epoch, the
identity check, and the exact offline thresholds. Each battle gate retains its
complete summaries, decisions, schedules, and replays.

## Smoke test

The smoke command exercises data selection, caching, and training without
spending time on battles. Skipping battle evaluation can never promote the
candidate; `selected_checkpoint.txt` stays on the champion.

```bash
python -m pokemon_battler.residual_pipeline \
  --output-dir outputs/champion-residual-smoke \
  --teacher-train-rows 128 \
  --teacher-validation-rows 64 \
  --replay-train-rows 256 \
  --replay-validation-rows 128 \
  --epochs 1 \
  --skip-battle-evaluation
```

## Live and public inference

The existing live runtime detects `residual_head.safetensors`, obtains the base
Qwen interaction outputs, and applies the correction before selecting an action.
No teacher or search engine is used at inference. Action values remain the
champion's frozen diagnostic values and are not blended into selection.
Local and paired-battle summaries record `policy_architecture: residual`, so a
run cannot silently look like it exercised the correction when it loaded only
the interaction policy.

The old statewise PPO updater refuses a residual checkpoint because it only
knows how to update the interaction policy. Allowing it to run would silently
discard the residual that actually chose the rollout actions.
