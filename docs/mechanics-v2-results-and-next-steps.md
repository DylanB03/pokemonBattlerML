# What mechanics-v2 taught me, and what I changed next

I got a real improvement from mechanics-v2, and the run also made the current
ceiling easier to see. The selected `final/` checkpoint reached 2,143 correct
decisions out of 5,000 validation rows: 42.86% top-1 action agreement. Top-2
was 64.78%, top-3 was 78.78%, and mean reciprocal rank was 0.6265. On the fixed
1,024-row validation subset used during training, the best checkpoint reached
41.8945%, compared with 36.5234% for the earlier candidate-head run. That is a
5.37-point gain with the same Qwen2.5-0.5B base model.

That gain showed me that the previous 36% result was not the inevitable limit
of a 0.5B model. Better candidate information and fewer representation
collisions produced substantially better decisions without a larger language
model.

I still thought 0.5B parameters and rank-16 LoRA might impose a capacity
ceiling. Long-horizon planning, hidden-set inference, and unusual
interactions are difficult tasks. The completed experiment did not isolate
that ceiling because the data, scorer, and training target still left important
performance on the table. I decided not to pay for a larger model until I had
tested those limitations.

## What the result actually says

| Metric | Mechanics-v2 final |
| --- | ---: |
| Exact action agreement | 42.86% |
| Top-2 agreement | 64.78% |
| Top-3 agreement | 78.78% |
| Candidate NLL | 1.5452 |
| Move-versus-switch accuracy | 83.72% |
| Ordinary-move accuracy | 50.78% |
| Switch accuracy | 31.16% |
| Tera-move accuracy | 3.57% |

The model usually understands the broad decision family, but it is much weaker
at choosing the exact switch and almost never selects a Tera action. There were
112 Tera targets in the 5,000-row evaluation, while the model made only five
Tera predictions. Four were correct. That looks less like random inability and
more like a policy that learned a very conservative Tera prior from an
imbalanced one-label objective.

The loss did not collapse, but it did keep improving. The rolling training-loss
mean fell from about 1.86 over steps 1-500 to 1.55 over steps 6,001-8,000.
Validation top-1 accuracy peaked earlier while validation NLL continued to
decline. More updates to the same system would probably make its probability
ranking a little smoother; there is no evidence that another epoch would
produce a comparable jump in exact decisions.

## The data problem I found after the run

I found that the completed run used the existing `data/gen9ou-dev` files, which
predate the current preparation schema. A full scan found that all 286,059
training rows, all 28,556 validation rows, and all 11,662 test rows lack:

- `schema_version: 2`;
- four-turn `recent_move_history`;
- accumulated `opponent_revealed_pokemon` information;
- preparation-time `player_remaining` and state turn metadata;
- `legal_mask_quality`, including exact and zero-PP-aware mask provenance.

The loader reconstructs turn index and remaining Pokémon for these legacy rows,
so those two values were not completely absent at training time. It cannot
reconstruct prior opponent reveals or four-turn history from an isolated row.
It also cannot tell evaluation apart by exact, PP-aware, and merely recoverable
legal masks.

I still count the 42.86% result, but I could not reuse this prepared dataset for
the next long run. The current preparer already implements the missing fields,
so I rebuilt the data in a new directory before the next long run:

```bash
pokemon-prepare \
  --input data/raw/metamon/gen9ou.tar.gz \
  --output-dir data/gen9ou-dev-schema2 \
  --format gen9ou \
  --min-rating 1600 \
  --outcome both \
  --split-mode chronological \
  --validation-start 2026-01-01 \
  --test-start 2026-04-01 \
  --sample-rate 0.02 \
  --seed 42
```

Keeping a new directory preserves the old rows, caches, and checkpoint. The
deterministic split and sample settings also make the new result as comparable
as the source archive permits.

## The performance changes I chose

These are ordered by how directly they address a measured limitation, not by
how novel they sound.

### 1. Prove the v2 path can memorize before another full run

Run a mechanics-v2 overfit gate on 128 examples and require at least 95% exact
agreement. This isolates wiring, optimization, LoRA, and scorer capacity from
generalization. If the exact architecture cannot nearly memorize 128 rows,
more data and more epochs are the wrong response.

The existing memorization gate predates the final v2 representation, so I
needed to repeat it for this actual input and head.

After preparing `data/gen9ou-dev-schema2`, the exact gate is:

```bash
pokemon-mechanics-cache \
  --data-file data/gen9ou-dev-schema2/train.jsonl

pokemon-train \
  --model Qwen/Qwen2.5-0.5B \
  --train-file data/gen9ou-dev-schema2/train.jsonl \
  --train-mechanics-cache data/gen9ou-dev-schema2/train.mechanics-v2.npy \
  --output-dir outputs/mechanics-v2-schema2-overfit-128 \
  --objective mechanics-head \
  --prompt-format mechanics-v2 \
  --method lora \
  --local-files-only \
  --load-in-4bit \
  --dtype auto \
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

pokemon-evaluate \
  --adapter outputs/mechanics-v2-schema2-overfit-128/final \
  --data-file data/gen9ou-dev-schema2/train.jsonl \
  --mechanics-cache data/gen9ou-dev-schema2/train.mechanics-v2.npy \
  --max-examples 128 \
  --scoring auto \
  --prompt-format auto \
  --local-files-only \
  --load-in-4bit \
  --output reports/mechanics-v2-schema2-overfit-128.json
```

### 2. Train on the current prepared schema

Regenerate the 2% dataset as shown above, rebuild its mechanics caches, and run
the same experiment once. Report accuracy separately for exact, PP-aware, and
recoverable masks. The added history and opponent reveals should be treated as
a controlled data change, not bundled with a new architecture in the first
comparison.

Once the gate passes, the controlled full command is:

```bash
python -m pokemon_battler.experiment \
  --train-file data/gen9ou-dev-schema2/train.jsonl \
  --validation-file data/gen9ou-dev-schema2/validation.jsonl \
  --output-dir outputs/mechanics-v2-schema2-1epoch
```

### 3. Replace independent candidate scoring with interaction

The current `MechanicsHead` repeats one final Qwen state vector 13 times and
scores each candidate with the same MLP. Candidate A never attends to candidate
B. The network can compare their final scalar logits through softmax, but it
cannot learn an explicit interaction such as “this switch is worthwhile only
because every attack is worse,” or reason jointly over the full team and all
candidate actions.

I decided that the next head needed to represent team members and legal
candidates as a small set of tokens, then use two to four lightweight attention
layers before producing masked logits. A global token can retain the Qwen state
vector initially. This is the kind of interaction that a Set Transformer is
designed to model; its original paper describes attention over set elements
without imposing an arbitrary order
([Lee et al., 2019](https://arxiv.org/abs/1810.00825)). Such a head can be tens
of millions of parameters, fit beside the current 0.5B backbone, and is not a
model upgrade.

### 4. Make action family an explicit decision

Use a hierarchical policy:

```text
P(action | state) = P(move, switch, or Tera | state)
                    x P(exact action | chosen family, state)
```

Keep ordinary legal-action cross-entropy as the primary loss, then add an
auxiliary family loss and cap its weight. This gives rare Tera decisions a
usable training signal without making their 2.4% training frequency look like
one third of real decisions. Evaluate on the natural distribution and report
both family recall and precision. A quick `family-balanced` run is a useful
ablation, but it is not as clean as a proper hierarchical head.

### 5. Fuse mechanics before the final scalar scorer

Qwen currently finishes encoding text before it sees any of the 207 numeric
candidate values. The shallow head can combine the final state vector with
damage and matchup values, but Qwen's internal attention cannot reason over
those numbers. Project candidate mechanics and Pokémon state into learned
tokens before the interaction layers. Keep compact names as categorical
identity, not as a replacement for calculated mechanics.

This also creates a clean capacity test: compare a structured policy with no
Qwen token, one frozen Qwen global token, and the LoRA-adapted Qwen token. If the
structured model matches the hybrid, the language backbone is adding cost
rather than useful state information.

### 6. Add a value objective, then optimize decisions for outcomes

The current loss asks one question: “which action did this player choose?” It
does not ask whether the battle was won, whether several actions were nearly
tied, or whether the recorded choice was a mistake. Both winning and losing
perspectives currently receive equal one-hot imitation targets.

Add a state-value head trained on the recorded outcome and share its structured
state encoder with the policy. After that head is calibrated, move from plain
behavior cloning to conservative advantage-weighted imitation or offline RL,
using trajectory outcome and simulator-derived changes in HP, status, and
faints. Do not simply discard every losing trajectory: many decisions in a
lost game are good, and wins-only filtering changes the state distribution.

This direction is supported by the project whose data this repository uses.
Metamon progresses from imitation learning to offline reinforcement learning
and self-play rather than treating exact human-action cloning as the final
objective ([Hu et al., 2025](https://arxiv.org/abs/2504.04395)). Its reported
models are also smaller than 0.5B. Its older-generation settings are not
directly comparable with unrestricted Generation 9 OU, but the result is still
useful evidence that architecture, state representation, and objective can
matter more than language-model scale.

### 7. Use the policy as a proposal inside shallow search

Offline action agreement will always penalize plausible alternatives to the
single recorded human action. The real target is battle win rate. Once a local
Showdown adapter and value model exist, let the policy propose the top few legal
actions, simulate one to three plies against an opponent model, and use the
value head to score leaves. This can improve play without making every live
decision run a large language model over every branch.

This is also why the often-cited 49% PokeLLMon number is not directly comparable
to 42.86% action agreement. PokeLLMon reports live ladder win rate and combines
in-context feedback, external knowledge, and consistent action generation
([Hu et al., 2024](https://arxiv.org/abs/2402.01118)). PokéChamp more directly
demonstrates the value of search: it combines LLM action sampling, opponent
modeling, value estimation, and minimax tree search
([Zhang et al., 2025](https://arxiv.org/abs/2503.04094)). Those systems measure
winning games, not matching one replay label.

### 8. Distill expensive decisions back into the small policy

Search can generate a distribution over several actions instead of one hard
label. Train the small policy to reproduce those soft values offline. This
keeps the 0.5B runtime while teaching it decisions that incorporate future
consequences. It also reduces the irreducible noise caused by several actions
being reasonable in the same state.

## How to determine whether 0.5B is the real limit

The model-size question needs controlled evidence:

1. Require the current v2 stack to overfit 128 rows.
2. Compare the structured interaction head with and without Qwen on identical
   rows and masks.
3. Compare frozen-Qwen, current LoRA, and a less constrained state encoder while
   keeping the head and data fixed.
4. Only if all three pass and validation still saturates should model size be
   the next variable.

If the v2 stack cannot memorize, there is a pipeline or capacity bottleneck. If
the structured model beats the hybrid, the current language representation is
the bottleneck. If Qwen helps but both plateau after the data and objective are
fixed, 0.5B capacity becomes the strongest explanation.

## The order I chose

I staged the next work so each expensive run answered one question:

1. Preserve the current checkpoint and reports.
2. Regenerate the 2% dataset with schema 2.
3. Run the 128-row v2 memorization gate.
4. Repeat mechanics-v2 on the fresh rows as the controlled data test.
5. Build the candidate/team interaction head with hierarchical family loss.
6. Add the shared value head and evaluate outcome calibration.
7. Connect a fixed-team simulator benchmark and add shallow search.
8. Distill search values into the small policy.

I decided not to spend another 20-hour run on learning-rate changes, a longer
copy of the same epoch, RAG prose, or a blind model upgrade. The result was good
enough to justify the project, but not good enough to leave the data,
independent scorer, and imitation-only objective unchanged.
