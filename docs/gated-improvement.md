# Gated Qwen improvement

The previous distillation pipeline proved that the interaction head could move
toward Foul Play. It did not prove that the resulting bot won more games. Round
zero took 22 hours, teacher top-one agreement rose from 43.0% to 53.8%, and the
paired held-out result was 12 wins for the candidate versus 9 for the champion.
The 95% interval still crossed zero. Continuing the same loop would have made
each round slower because every round retrained on a larger aggregate.

This runner is deliberately different. It spends a small amount of time trying
to disprove an idea before giving that idea another long training run.

## Run it

From the repository root, in the already activated Python environment:

```bash
python -m pokemon_battler.gated_pipeline \
  --output-dir outputs/qwen-gated-v1
```

Use a new output directory for every attempt. The runner refuses to write into
an existing directory and never replaces a source checkpoint.

The defaults use:

- champion: `outputs/public-learning/positive-winrate-1000/batch-005/candidate`;
- diagnostic candidate: `outputs/qwen-dagger-v1/round-00/candidate`;
- teacher data: round 00 and round 01 from `outputs/qwen-dagger-v1`;
- human replay: `data/gen9ou-interaction-v1/train.jsonl` and `validation.jsonl`;
- opponent pool: `examples/opponent-pools/gen9ou-foul-play.txt`;
- fixed deployment team: `examples/teams/gen9ou-balance.txt`.

Override any of those paths explicitly when a newer checkpoint or teacher
collection exists. Repeating `--teacher-data` replaces the two default teacher
files rather than adding to them.

## What the command does

### 1. Measure the teacher ceiling on held-out teams

Three enemy compositions are reserved before any selection or training. Foul
Play controls the fixed deployment team against an independent Foul Play
opponent for 50 games. This serves two purposes: it estimates how well the
teacher can do in this exact fixture, and it creates genuinely unseen teacher
states for offline validation.

If Foul Play itself performs poorly, a student trained to imitate it cannot be
expected to solve that fixture. The result is recorded rather than silently
treated as a training failure.

### 2. Ablate inference before training

The champion is evaluated once with submitted-order preview and no Q-logit
blend. The existing candidate is then evaluated on the identical schedule in
up to four configurations:

- policy only;
- policy plus the old `0.35` Q-logit blend;
- learned preview plus policy;
- learned preview plus the Q blend.

This shows whether the candidate head improved while deployment-time additions
hid the gain, or whether those additions were helping. The training warm start
uses the candidate only when its policy-only result is at least as good as the
champion on the paired diagnostic schedule.

### 3. Select bounded training data

The runner does not concatenate every row forever. It selects at most 8,000
teacher decisions, prioritizing states where a student action disagreed with a
confident teacher and where their distributions were far apart. It targets a
60/35/5 attack/switch/Tera mix when enough examples exist, balances enemy teams,
and caps each battle at 24 selected turns.

Human-replay rehearsal is a deterministic 4,000-row sample. Another 2,000 rows
come from the existing battle-separated replay validation file. Held-out Foul
Play validation remains separate from both.

### 4. Cache frozen Qwen once

Qwen and its existing LoRA adapter remain frozen during these head experiments.
The runner therefore stores Qwen's final state vector and all structured input
tensors once for teacher train, teacher validation, replay train, and replay
validation. Every later epoch reads those tensors directly. It no longer pays
for two Qwen forward passes on every optimizer step.

The cache is tied to one starting checkpoint. Do not reuse it with a checkpoint
whose Qwen or LoRA weights differ.

### 5. Prove the head can memorize

Before the real variants run, the head receives 256 cached teacher examples for
25 short epochs. It must reach at least 85% teacher top-one agreement on those
same examples. Failure means the model, cache, targets, or optimizer is wired
incorrectly; the runner stops without starting the larger experiment.

### 6. Compare objectives on the same tensors

Three heads start from the same checkpoint and see the same selected rows:

- soft teacher policy only;
- soft policy plus the move/switch/Tera family objective;
- policy, family, and relative MCTS Q ranking.

Relative Q ranking asks which legal action has the better teacher value. It does
not assume that Foul Play's raw values are calibrated probabilities, and the Q
head is not blended into deployment scores during this experiment.

Each variant must improve held-out teacher agreement by at least three points,
reduce teacher KL, and lose no more than two points of replay action accuracy.
If none pass, the command stops and retains all reports and checkpoints.

### 7. Run one paired battle gate

Only the strongest eligible offline variant plays the final 100-game suite.
Candidate and champion both use policy-only inference and submitted-order
preview, so the result measures the trained turn head rather than a second
inference change. The exact enemy-team schedule is paired, and promotion still
requires a positive win-rate delta whose bootstrap interval is not strongly
negative.

## Outputs

```text
outputs/qwen-gated-v1/
  manifest.json
  selected_checkpoint.txt
  01-expert-ceiling/
  02-inference-ablation/
  03-selected-data/
  04-frozen-cache/
  05-overfit-gate/
  06-objective-variants/
    policy-only/
    policy-family/
    policy-family-q-ranking/
  07-paired-evaluation/
```

`manifest.json` is updated after every gate and records `running`, a specific
stop reason, `complete-promoted`, `complete-rejected`, or `failed`. A stopped
run is useful evidence, not an incomplete training job.

## Time and resource expectations

On the current RTX 3060 Ti and 16-thread CPU, budget roughly three to six hours
for the default run. Local Foul Play diagnostics and the one-time Qwen cache are
the expensive stages. Cached head epochs should be minutes rather than tens of
hours. A failed memorization or offline gate ends earlier.

The cache batch size defaults to four for 8 GB VRAM. Lower
`--cache-batch-size` to two after an out-of-memory error. Raising it is safe only
after measuring peak allocation. `--concurrent-games 4` already uses battle
parallelism; increasing Foul Play's internal threads at the same time will
oversubscribe this machine.

## What this does not build yet

This is the fastest reliable test of the current statewise architecture. It
does not add full-battle recurrent memory, perform public online learning, or
unfreeze Qwen. If a correctly wired cached head passes its offline gates but
cannot improve paired battles, the next justified change is trajectory memory
and long-horizon value training—not another larger pass over the same isolated
states.
