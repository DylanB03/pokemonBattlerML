# I Was Training a Pokémon Policy to Write `A4`

*The model's mediocre results forced me to rethink the loss, the action
representation, and what I was willing to call a result.*

*Status: August 2026. I stopped the candidate-ranking run at step 5,000 and
built the mechanics-conditioned experiment that follows it.*

The unfine-tuned model's first answer to a battle prompt looked like this:

```text
<legal_actions_counts>
<A0> 0
<A1> 0
<A2> 0
<A3> 0
<A4> 0
<A5> 0
</legal_actions_counts>
```

Plausible XML, useless decision. It was a good preview of the work ahead.

I am building a Generation 9 OU battle policy with
`Qwen/Qwen2.5-0.5B`. The eventual product I have in mind is bigger than a bot.
I want a post-game coach that can show its action preferences, reconstruct what
was known on each turn, and explain questionable decisions without pretending
that policy likelihood is the same thing as win probability.

The first problem is narrower. Given the public state of a battle, predict the
action recorded in a strong player's replay. That gives me a supervised policy
I can test before connecting it to Pokémon Showdown or trying to estimate the
value of moves that were never played.

## Trying to write the resume bullet exposed the weak spots

The project changed while I was trying to describe it for an ML internship. I
wanted a performance bullet, and the obvious ones did not survive much
scrutiny.

The easiest claim was a “100% legal-action rate.” It collapsed as soon as I
looked at the evaluator. Illegal actions were removed before selection. The
system guaranteed legality; the model had not earned a perfect score.

One evaluation showed roughly 35% accuracy after I supplied the correct
move-versus-switch category. The precise name for it was within-type agreement.
Its random baseline depended on the number of choices of that type in every
state, so a vague one-in-four estimate would not work.

Selective vocabulary projection was already in the training loop, yet my local
VRAM measurements showed no useful difference. I had an implementation
hypothesis and no benchmark result.

The clean-looking training loss was suspect too. It had flattened around 0.6,
while the policy still made the wrong decision more than two-thirds of the time.

Writing around those details would have produced a better bullet and the same
weak model. I turned them into an experiment list: exact baselines, end-to-end
metrics, error slices, a direct action loss, and enough accounting to know how
much of the dataset the model had seen.

## The scope is intentionally small

I kept the 0.5B Qwen checkpoint. It fits the machine I have, and a bigger model
would let me spend more compute without proving that the pipeline made sense.

The base weights load in 4-bit NF4 with double quantization. LoRA adapters cover
the attention and MLP projections. I train one example at a time and accumulate
32 microbatches before each optimizer update. The setup fits on an 8 GB NVIDIA
GPU.

Pokémon also gives me a small output space. Metamon maps a turn to at most 13
universal actions:

- `A0` through `A3` are ordinary moves in alphabetical order.
- `A4` through `A8` are available switches in alphabetical order.
- `A9` through `A12` repeat the four moves with Terastallization enabled.

The IDs are positional. `A4` might mean switching to Corviknight in one state
and Great Tusk in another. If a teammate faints, the remaining switch IDs can
shift. The prompt needs to say what each ID means on that turn.

## Preparing replay data without leaking battles

The source Metamon archive is about 20.4 GB compressed. My preparation code
reads JSON, JSON-LZ4, tar, and compressed tar files as streams, so I never have
to unpack the whole archive.

For the development split, I kept battles rated 1600 or higher and sampled 2%
of eligible turns. I split by date:

```text
train       before 2026-01-01
validation  2026-01-01 through 2026-03-31
test        2026-04-01 and later
```

| Split | Battles | Decisions |
| --- | ---: | ---: |
| Train | 127,229 | 286,059 |
| Validation | 12,747 | 28,556 |
| Test | 5,306 | 11,662 |

Turns from the same battle stay together. Splitting turns independently would
put adjacent states from one game into train and validation, which is about the
easiest way to manufacture a flattering result on replay data.

There is still a replay-to-live gap. Spectator replays do not always expose the
exact request mask a live player saw. I can recover moves, available switches,
forced switches, and Terastallization candidates, but mechanics such as
trapping, disabled moves, and Choice locks can require information absent from
the replay. Offline predictions use the recoverable candidate set. I will need
the live simulator mask before reporting a real legality rate.

## My first objective optimized the spelling of the answer

The original model used causal language modeling. I serialized the state and
the legal action mapping, then trained Qwen to generate a string such as `A4`
followed by EOS.

All prompt labels were `-100`. Only the action string and EOS contributed to
loss. I also used Qwen's selective-logits argument to avoid projecting every
prompt position into its 151,936-token vocabulary. A typical prompt was about
2,000 tokens long, while only the final few positions needed vocabulary logits.

That optimization removes unnecessary work in the language-model head. It does
not remove transformer attention or MLP work. On this model, those activations,
the weights, optimizer state, temporary kernels, and PyTorch's caching allocator
can dominate the number shown by `nvidia-smi`. The code now supports full and
selective projection as separate benchmark modes. I will claim a memory or
throughput improvement when I have measurements from identical batches in
separate processes.

Before running larger experiments, I forced the model to memorize 128 training
examples. It reproduced 126 of them, or 98.4%, after 400 updates. The test was
deliberately easy. Generalization remained untested. The 128-row test checks the
connections: labels reached the loss, LoRA adapters received gradients, and
saved checkpoints recovered what they learned.

I also scanned for exact duplicate states. There were 15 and none had
conflicting labels. Broken plumbing and obvious label collisions moved down the
suspect list.

## The first real runs plateaued

I completed 2,000-update runs at `1e-4` and `2e-4`, then evaluated an available
checkpoint from a `4e-4` run. All three used the generative action-string
objective. On the same 500 validation examples, I got:

| Learning rate | Final training loss | Accuracy | Move accuracy | Switch accuracy |
| --- | ---: | ---: | ---: | ---: |
| `1e-4` | 0.594 | 29.4% | 35.8% | 18.9% |
| `2e-4` | 0.591 | 29.0% | 37.4% | 15.3% |
| `4e-4` | n/a | 21.0% | 25.8% | 13.2% |

The two completed loss curves were nearly identical. Raising the learning rate
made the available result worse. I had a hard time believing that another small
learning-rate adjustment would explain a plateau that started so early.

The run budget was one problem. An effective batch of 32 over 2,000 updates is
64,000 examples, or 22.4% of the 286,059-row training set. The CLI said
`epochs=100`, but `max_steps=2000` ended the run during the first pass. The
cosine schedule also treated 2,000 as its entire horizon and decayed the rate to
zero.

The loss itself hid another problem. Qwen tokenizes `A0` through `A9` as `A`
plus one digit. `A10` through `A12` use `A` plus two digits. The trainer appends
EOS, so it averages loss over three or four supervised tokens. The repeated `A`
and EOS are easy. A displayed loss of 0.6 can coexist with a much larger loss on
the digit that separates one decision from another.

The prediction histogram was worse than the scalar loss suggested. The `1e-4`
model favored common move slots and never predicted `A9` through `A12` on those
500 examples, despite Terastallized actions appearing in the targets. Switch
accuracy was 18.9%. The model had learned the answer format and some action
frequency, but not enough of the decision.

## A legal mask cannot rank the legal actions for me

The mask removes illegal actions from the distribution by setting their logits
to negative infinity. They receive zero probability and cannot win the argmax.
I use the same rule during training and evaluation for the new action heads.

The mask handles the candidate constraint. It cannot tell the model whether to
click Earthquake, preserve a win condition, switch out of a bad matchup, or
commit Terastallization. “100% legal” belongs in the implementation description,
next to input validation and checkpoint loading. The learned result is the
ranking inside that mask.

I first replaced the language-model loss with a fixed 13-way policy head. It
removed the tokenization problem and trained cross-entropy directly over legal
action IDs. The head still assigned one output neuron to `A4`, even though `A4`
can name a different Pokémon on every turn.

The candidate head attaches scores to action descriptions. Each legal
action gets a short description containing the move and Terastallization flag
or the switch target. I take the transformer hidden state at that description
and pass it through one shared MLP. The loss is:

\[
L=-\log\frac{\exp(s(a_{\text{target}}))}
{\sum_{a\in A_{\text{legal}}}\exp(s(a))}
\]

Every legal candidate uses the same scorer. Training asks for a relationship
between the battle state and the candidate's contents. I randomize candidate
order during training so the model cannot depend on one presentation order as
easily. The LoRA-adapted transformer stays trainable, but the 151,936-output
language-model head is no longer part of this objective.

## Cutting the prompt nearly in half

The first prompt format repeated field names, default values, and XML-style
markup. On 25,000 training examples it used a median of 2,057 tokens. The 99th
percentile was 2,463 and the longest prompt was 2,511.

I wrote a versioned compact serializer and kept the battle facts and candidate
descriptions. The median dropped to 1,157 tokens, with a 99th percentile of
1,406 and a maximum of 1,460. The median reduction is about 44%, and it cuts
attention and MLP work throughout the transformer.

I added turn number and remaining team size to old prepared rows at load time.
The version-two preparation path can also carry recent move history, opponent
Pokémon and moves revealed on earlier turns, conservative zero-PP filtering,
and an exact legal mask when the source provides one. Opponent memory is built
while walking the replay forward. An earlier turn never receives a move or
Pokémon revealed later.

The full opponent-history and PP-aware changes require regenerated data. The
current run uses the existing JSONL files with the backward-compatible turn and
team-count additions. I would rather state that limitation than quietly imply
that every schema improvement is already present in the running experiment.

## Making the next run readable

The evaluator now selects rows by a deterministic hash. Hash sampling replaces
head-of-file selection, so every checkpoint sees the same examples.

For action-head checkpoints, one report includes action NLL, top-1 through
top-3 accuracy, mean reciprocal rank, move-versus-switch accuracy, move,
switch, and Terastallization slices, entropy, margins, and the full target and
prediction counts. Checkpoint selection uses validation accuracy, with NLL as
the tiebreaker. At the end, the runner evaluates both `best/` and `final/` on
the same 5,000 rows.

I added uniform and training-frequency baselines, base-Qwen evaluation, and a
small feature-hashed action network. It tests whether a 500M-parameter causal
model earns its cost on a task with at most 13 actions. If structured features
tie Qwen, the language model has not earned that cost yet.

The training loop now prints the number of examples and tokens processed, the
fraction of a dataset pass, examples and tokens per second, total gradient norm,
separate head and LoRA gradient norms, clipping, nonfinite gradients, and peak
allocated VRAM. Short debug runs no longer compress the scheduler unless I ask
them to. The normal cosine schedule spans one full pass and keeps a 5% learning
rate floor.

With 286,059 examples and an effective batch of 32, the new run lasts about
8,940 optimizer updates. It starts with one command:

```bash
.venv/bin/python -m pokemon_battler.experiment \
  --output-dir outputs/candidate-compact-1epoch
```

The runner trains, evaluates `best/` and `final/`, and writes the comparison to
`run_summary.json`.

## The higher loss was honest, but the run still flattened

Around step 200, the candidate-head training loss was about 1.8. The old loss
was already near 0.6 at a similar point. Comparing them directly would have
repeated the mistake that led to the redesign.

The old number averaged the action decision with easy `A` and EOS tokens. The
new number is one cross-entropy over every legal candidate. Across the training
set, uniform legal selection has an average NLL of 2.127. A loss of 1.8 gives a
geometric mean target probability of about 16.5%; the uniform-NLL reference is
11.9%.

Step 200 was inside the roughly 268-step warmup and covered about 2.2% of one
dataset pass. The loss had moved below uniform. That was encouraging, but it did
not settle whether the candidate representation would keep improving.

It did improve, then flattened. At step 5,000 the best fixed 1,024-row
validation result was 374 correct decisions:

| Metric | Candidate-head result |
| --- | ---: |
| Net top-1 action agreement | 36.5234% |
| Validation candidate NLL | 1.6630 |
| Dataset pass completed | about 55.9% |

The net accuracy is the fraction of all decisions where the exact recorded
action was ranked first. It is not the earlier within-type number where the
correct move-versus-switch category was supplied. That distinction makes the
result meaningful. It also does not make it a battle win rate.

Training loss spent thousands of updates moving between roughly 1.60 and 1.75.
Individual log points were noisy, but the validation accuracy had also stopped
moving enough to justify the remaining compute. I stopped at 5,000 instead of
turning “finish the epoch” into an objective of its own. This run had consumed
about 160,000 examples. Another small learning-rate adjustment was not the most
convincing next idea.

The older 29.4% generative result used 500 validation examples, while the 36.52%
result used a deterministic 1,024-row sample. I therefore treat the new number
as a reference, not a clean 7.1-point head-to-head improvement. The final runner
will evaluate selected checkpoints on the same 5,000 rows.

## I stopped asking Qwen to rediscover the type chart

The candidate head fixed the loss, but its input still represented moves as
names. A 0.5B language model had to learn that a name implied a type, power,
accuracy, secondary effect, stat change, or switch cost before it could learn
whether the action fit the situation.

Those facts are not language. They are deterministic game mechanics.

I built a second candidate branch with 97 numeric values per action. It includes
type effectiveness, STAB, priority, PP, expected hits, approximate damage range,
healing, recoil, status chances, stat-stage changes, hazard interaction, switch
HP after entry, and known offensive and defensive matchup pressure. The tensor
has shape `13 x 97`; it goes straight into a small MLP and adds no prompt tokens.

The language model still has a job. It reads one compact state containing the
Pokémon, team context, field, HP, items, abilities, base stats, statuses, and
boosts. It produces a state vector. A shared scorer combines that vector with
each candidate's mechanics vector and assigns the legal logits.

I removed move lines, previous move names, and candidate descriptions from this
prompt. On the first 100 training rows, its median was 466 tokens. The old
compact candidate prompt was 1,211 on those same rows. Adding structured
mechanics therefore cut the text by about 61% in that check instead of tripling
it.

Move names still appear inside the deterministic preprocessing step because
they are the key used to retrieve structured move data. They are discarded
before model input. Damage is explicitly approximate: spectator replays do not
contain EVs, IVs, natures, or every temporary modifier. I would rather provide a
consistent pressure estimate and an availability flag than call an invented
number exact.

This is not RAG. There is no generated type-chart paragraph for Qwen to parse.
The model receives floats. The same feature builder can later run on live battle
state, where simulator-backed damage can replace the approximate columns.

## The MLP is an ablation, not an abandonment of language models

I also added a mechanics-only policy: a small multilayer perceptron that scores
the same `13 x 97` tensor with the same legal mask and no Qwen representation.
It answers one useful question. If it matches the hybrid, Qwen is not adding
enough strategic context to earn its compute. If the hybrid wins, the gap is the
value of the learned state representation.

That comparison does not force the project away from language models. It gives
the language model a falsifiable role.

The next hybrid run starts with one command:

```bash
.venv/bin/python -m pokemon_battler.experiment \
  --output-dir outputs/mechanics-v2-1epoch
```

The command builds reusable memory-mapped mechanics caches, trains the hybrid,
evaluates the best and final checkpoints, and writes the report. It now stops
after four validation checks without at least a 0.2-point accuracy gain. The
best checkpoint still records every exact gain, so early stopping only controls
the compute budget.

I am still keeping the 0.5B model. I have also left active learning, continual
learning, simulator RL, resume support, and model upgrades out of this phase.
The next experiment changes the representation of facts the current model was
being asked to relearn.

## I changed the representation again after measuring its collisions

The first mechanics version was too aggressive about removing identity. I had
treated move names as a shortcut I should eliminate, then checked the actual
vectors across the training split. In 2,880 rows, at least two legal actions had
exactly the same 97 values. The recorded action was inside one of those tied
groups 780 times. Rest and Sleep Talk could look identical. So could Reflect
and Light Screen, Spikes and Stealth Rock, or two attacks whose important
difference lived in a custom callback.

No optimizer can separate two candidates when both branches receive identical
inputs. That made this a representation bug, not another learning-rate debate.

I replaced the single numeric summary with a hybrid structured candidate. It
now has 207 numeric values for reusable mechanics and 32 categorical fields
with learned embeddings for exact moves, species, items, abilities, types,
statuses, effects, field state, and recent move history. I also restored compact
move-name lists to the state prompt. The numeric branch still supplies damage,
type effectiveness, status chances, screens, hazards, speed order, and the
other facts I do not want Qwen to infer from a name. Identity is there for the
exceptions that cannot honestly be compressed into one flag.

This costs some prompt length and a larger feature cache, but it does not triple
the battle prose or ask the model to perform retrieval. It gives the 0.5B model
both kinds of signal: numbers for generalization and identities for special
cases. The old schema remains loadable, and the new run writes to its own
mechanics-v2 caches and output directory.

If the mechanics-conditioned checkpoint improves on the 36.52% reference, I
will compare it with the mechanics-only ablation and take the selected policy
into reproducible Showdown battles against fixed opponents. Replay agreement is
a useful gate. Real games remain the test that matters.

## The run improved, but it exposed the next ceiling

The mechanics-v2 run ended at step 8,000 after 20 hours and 48 minutes. Its
selected final checkpoint got 2,143 of 5,000 validation decisions exactly
right: 42.86%. Top-2 was 64.78% and top-3 was 78.78%. On the fixed 1,024 rows I
had used to judge the earlier run, the best checkpoint went from 36.52% to
41.89%. I did not change the 0.5B base model to get that gain. I changed what
the policy could see about each action.

That answered one question and left a harder one. A 0.5B model may simply be
too small for some of the strategy I want. I cannot honestly conclude that
from this run, though. The same model had just gained more than five points
from a representation change, and the remaining failures were not evenly
distributed. It got 50.78% of ordinary moves, 31.16% of switches, and only
3.57% of Tera moves. It predicted Tera five times in 5,000 decisions despite
112 Tera targets. Four of those five predictions were right. The model had
learned to avoid the rare family, not merely to choose randomly inside it.

I also found that I had trained on older prepared rows. Every row in the train,
validation, and test files predates the current preparation schema. The loader
could reconstruct turn number and remaining Pokémon, but it could not recreate
four turns of move history, accumulated information about revealed opponents,
or the quality of the legal-action mask from one isolated row. The code to
write those fields already exists. I had not regenerated the dataset after
adding it.

The current scorer is another limit. It gives Qwen the compact text, takes one
final state vector, repeats that vector for all 13 candidates, and scores each
candidate independently with an MLP. The candidates meet only in the final
softmax. The mechanics values also arrive after Qwen has finished encoding the
state. That is efficient, but it is a weak way to represent the relationship
between a full team, an opponent, and several mutually exclusive decisions.

My next architecture will treat Pokémon and legal actions as a small set of
interacting tokens. A lightweight attention block can compare candidates and
team members directly, while the current Qwen vector remains one optional
global input. I also want a hierarchical move/switch/Tera objective and a value
head trained on battle outcome. Exact human-action cloning is a useful first
gate, but it cannot tell the difference between copying a losing choice and
choosing an alternative that wins.

I am still not treating a model upgrade as the default answer. First I will
regenerate the dataset, prove that the exact v2 stack can memorize 128 rows,
and compare the structured interaction policy with and without Qwen. If the
small model still helps and both versions plateau after the data and objective
are fixed, then model capacity becomes a much stronger explanation. Until
then, buying more parameters would hide the diagnosis.

The concrete experiment order and the comparison with published Pokémon agents
are in
[What mechanics-v2 proved, and what should change next](mechanics-v2-results-and-next-steps.md).
