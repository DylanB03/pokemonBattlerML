# Pokémon Battler SFT Harness

This repository fine-tunes a small causal language model to choose actions in
Generation 9 OU Pokémon battles. It converts Metamon replay trajectories into
turn-level examples, trains `Qwen/Qwen2.5-0.5B` with QLoRA, and evaluates only
actions that were legal in each recorded state.

The first objective is supervised behavior cloning: predict a strong human
player's recorded action from the visible battle state. Human-action agreement
is useful, but it is not the final measure of battle strength because several
actions can be reasonable. A later simulator evaluation must measure win rate
against fixed opponents.

See [ROADMAP.md](ROADMAP.md) for the planned progression from behavior cloning
to model-preference displays, replay review, counterfactual win-probability
analysis, regret estimation, and grounded coaching.

## Recommended experiment

This is the default protocol for one 8 GB NVIDIA GPU:

| Phase | Data and budget | Purpose | Required result |
| --- | --- | --- | --- |
| Environment check | Unit tests and cached model | Verify the installation | All tests pass and CUDA is visible |
| Data development set | 2% deterministic turn sample | Retain battle diversity at manageable cost | At least 50k train and 5k validation rows |
| Memorization gate | 128 rows, at most 400 updates | Detect broken labels, masking, LoRA, or loading | At least 95% accuracy on the same 128 rows |
| Hyperparameter search | Six runs, 2,000 updates each | Select learning rate, LoRA rank, and dropout | Best validation action accuracy |
| Main data | 10% deterministic turn sample | Scale the selected configuration | Reinspect lengths before training |
| Main SFT | Up to three passes or 20k updates | Produce the SFT checkpoint | Select `best/`, not automatically `final/` |
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
- Full fine-tuning, LoRA, and 4-bit QLoRA loading
- Best-checkpoint selection by validation loss
- Memory-bounded constrained evaluation that cannot select an illegal replay
  action

`prompt.md` is a human-readable example. The authoritative serializer is
`src/pokemon_battler/prompting.py`.

## 1. Install and verify the environment

Run all commands from the repository root:

```bash
cd /home/dylan/Gitrepos/allrepos/pokemonBattler
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .
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

Each run automatically saves the lowest-validation-loss adapter under `best/`.
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
outputs/qwen25-05b-main/best/   lowest sampled validation loss
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
