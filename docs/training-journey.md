# I Was Training a Pokémon Policy to Write `A4`

*The model's mediocre results forced me to rethink the loss, the action
representation, and what I was willing to call a result.*

*Status: August 2026. The strongest measured checkpoint is now the second
Metamon sidecar, `outputs/metamon-large-v2/04-candidate`. Across 1,000 frozen
public ladder games, it finished 502-498. This is the first positive public
result in the project at that scale.*

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
python -m pokemon_battler.experiment \
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
python -m pokemon_battler.experiment \
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
[What mechanics-v2 taught me, and what I changed next](mechanics-v2-results-and-next-steps.md).

## I built the next experiment as one pipeline

The raw archive already had ordered trajectories. I had been throwing most of
that structure away when I wrote isolated decision rows. The new preparer walks
each battle in order and writes a schema-3 row with stable rosters and the last
four observable transitions. It can use earlier unsampled turns as context, but
never a state after the target action.

I kept the 207 numeric mechanics values and 32 identity fields for each action.
The difference is where they meet. A small transformer now sees one global
token, up to twelve Pokémon, four history events, and thirteen action
candidates. A switch candidate is explicitly linked to the Pokémon it would
bring in. Legal actions are normalized through a move/switch/Tera hierarchy,
and an auxiliary head tries to predict the battle outcome from the global
state. Qwen remains a global residual rather than the only representation of
the battle.

I also removed the manual gap between data preparation and training. One
command scans the archive, builds version-checked caches, attempts to memorize
128 examples, and only then starts the full run. It preserves the gate model,
the best checkpoint, and the final checkpoint in separate directories. That
does not guarantee a better result. It does mean the next result will test the
architecture I intended, on the data schema I intended, without quietly
reusing the old isolated rows.

## I stopped treating a long run as evidence

The first Foul Play distillation round took 22 hours. It improved held-out
teacher agreement by more than ten points, but its paired battle result was
only 12 wins against the old model's 9. That was compatible with a real gain and
with noise. The next round restarted from the rejected checkpoint, added every
old row to a growing aggregate, and began paying for frozen Qwen again. More
compute was not making the experiment more convincing.

I replaced that loop with gates. I now test the learned preview and the Q-value
blend independently before training. I reserve unseen enemy teams for a Foul
Play ceiling and validation set. From the existing teacher games I keep a
bounded set of student disagreements, switching decisions, Tera decisions, and
representative attacks instead of allowing long battles and repeated teams to
dominate.

The largest mechanical change is simple: frozen means cached. Qwen produces one
state vector for each selected row once. Several interaction heads can then use
the same tensors, which turns an objective comparison from a collection of
day-long jobs into a short experiment. A 256-row memorization check runs first.
The real variants must improve unseen-team teacher agreement without losing
more than two points on held-out human replay. Only one surviving variant earns
a paired 100-game test.

This setup can still reject every model. That is intentional. A stopped run now
tells me whether the failure was wiring, generalization, replay forgetting, or
actual battle performance. If a statewise head passes the first three and still
does not win more games, I have a concrete reason to spend the next round of
work on full-trajectory memory instead of changing another learning rate.

## The gates rejected every later attempt

That last sentence described the experiment I wanted. The results were less
encouraging.

The public campaign played 500 rated games in five sets of 100. It finished
178-322, or 35.6%, with a 95% Wilson interval of 31.5% to 39.9%. No set had a
positive record. The five results were 39%, 28%, 38%, 40%, and 33%.

Those numbers also exposed how weak a single 100-game result is. The
`batch-001` checkpoint scored 28% and 38% in consecutive sets. `batch-003`
scored 40% and then 33%. The opponent pool and my rating changed during the
run, but that is exactly the point: the maximum of several ladder sets is not a
clean estimate of model strength. If I run until I see 51%, I am selecting for
luck unless the pooled record and rating trend rise too.

PPO produced three locally promoted candidates during that campaign. The last
one, `batch-005`, beat `batch-003` 24-16 in a 40-game local mirror match. It was
then selected as champion. The campaign stopped before `batch-005` played a
public set of its own, so I do not have a public win rate for it. Calling it the
best model means it is the winner of the checkpoint-selection chain, not that
it has proved itself on the ladder.

I tried three more ways to beat it:

- A Foul Play DAgger candidate won 12 of 100 held-out games while the incumbent
  won 9 of its 100. The paired evidence was too weak to promote it.
- The trajectory IQL experiment improved validation action accuracy to about
  44%, but the selected memoryless policy won 20 of 100 held-out games and
  `batch-005` won 21. The recurrent version did worse than the memoryless one in
  its architecture test.
- The final residual policy made a large improvement in soft teacher KL and a
  1.15-point improvement in teacher top-1 agreement. In the held-out battle
  test it tied `batch-005` exactly, 4-46 against 4-46. Its selection pointer was
  correctly left on `batch-005`.

The residual result was the clearest warning about my metrics. It changed the
top action on only 119 of 2,000 teacher-validation states, while making the
whole distribution less confident. That was enough to reduce KL from the
teacher, but the live policy chooses the top legal action. Better probabilities
below rank one did almost nothing to the deployed behavior. I had improved a
loss without improving the decisions that played the games.

## Why this is failing

The main failure is that I trained several proxies for winning and kept hoping
that one of them would turn into win rate.

Human action cloning asks, “What did this player choose?” Foul Play
distillation asks, “What did this search policy choose in this local state?”
PPO on a small public batch asks, “Which sampled actions happened to precede a
win?” None of those questions is the same as, “Which legal action maximizes the
chance of winning this battle against this opponent?” They can be useful
training signals, but I treated movement in their losses as stronger evidence
than it was.

The public trace shows the practical result. After removing forced switches,
the policy made 9,435 attacks, 1,228 switches, and 54 Tera actions. That is
88.0% attack, 11.5% switch, and 0.5% Tera. Across the five batches, the attack
rate rose from 86.3% to 89.2%. The bot did switch, so the issue was not a broken
action mapper. It learned an extremely conservative distribution over the
strategically unusual actions. This matches the replay evaluation, where Tera
was rare and switch accuracy lagged attack accuracy. Cross-entropy rewards the
common attack class thousands of times and barely punishes a policy that almost
never takes a rare but decisive action.

The 13 legal outputs hide another hard problem. `A1` does not mean one move
across the dataset; it means the second move in the current sorted list. A
switch action also changes identity when the available roster changes. The
interaction head receives mechanics and identities for the candidates, but the
policy still has to infer long-term value from a compressed state. Immediate
damage is easy to learn. Preserving a win condition, scouting a set, timing
Tera, accepting a sacrifice, or switching now to avoid losing three turns later
is much harder.

The state is only partially observed. An opponent's unrevealed moves, item,
ability, Tera type, and plan matter. A strong player carries a distribution over
those possibilities and updates it throughout the battle. Four recent events
and one Qwen state vector are not a real belief state. The recurrent experiment
added memory, but the training data and objective did not force that memory to
represent hidden sets or opponent intent. Adding a GRU does not create the
missing supervision.

The public PPO loop is especially sample-starved. One batch has 100 terminal
win/loss labels spread over thousands of decisions. A loss does not say which
of twenty decisions caused it, and a win does not make every decision good.
The opponent population changes, the ladder rating changes, and action sampling
adds another source of variance. Updating a 0.5B policy from that signal can
move it without teaching it why it won. The local promotion gate then tests a
40-game mirror match with the same fixed team on both sides. It can detect a
large regression in that matchup, but it is too small and too narrow to prove
an improvement against varied human teams.

The fixed player team helped control one source of variance and let the policy
specialize. It also means the public-learning stage has mostly optimized one
team's action distribution. Public opponents brought varied teams, while local
promotions used the same team for candidate and champion. That mismatch made
promotion cheaper, but less predictive of the result I cared about.

Teacher distillation had a related mismatch. Foul Play can search a local
simulator state using mechanics code. The student gets the resulting action
distribution and tries to compress it into its policy. A few hundred games
cover a tiny fraction of team matchups, revealed sets, and turn sequences.
Reusing those rows more often increases fit to that sample; it does not create
new strategic coverage. Soft logits add information about the teacher's
alternatives, but the residual run showed that matching them can improve KL
while leaving the greedy move almost unchanged.

There is also a capacity limit, although the evidence does not isolate it
cleanly. Qwen2.5-0.5B has to compress a long structured state, hidden-information
clues, candidate identities, and history into a small representation. The
mechanics-v2 change improved exact action agreement from 36.52% to 41.89% on
the fixed comparison, which proves representation was a major bottleneck.
Later architectures then stalled around 44%. A larger model might reason over
the state better, but it would still inherit sparse rewards, biased data, and
weak evaluation. More parameters alone would make the same experiment slower,
not make its target correct.

## What I could conclude before the large offline run

At this point, I could conclude that the training recipe had not produced a
positive-win-rate ladder policy. The 500-game record was far enough below 50%
that it was not a close miss. The later teacher, trajectory, and residual runs
also failed to beat the selected public checkpoint under their own promotion
rules.

I could not conclude that 35.6% was the exact strength of every checkpoint. Two
runs of the same checkpoint differed by seven to ten points, and `batch-005`
had no public batch. I also could not say the 0.5B base model was the sole
cause. The strongest measured gain had come from better inputs, while the
strongest measured failures came from objectives and gates that did not match
deployed play.

Offline action accuracy is not a scaled version of win rate. A 44% policy is
not “close” to a published agent with a 49% battle win rate. Accuracy compares
one action with one logged action. Win rate evaluates an entire sequence and
allows many different good actions. A policy can have lower imitation accuracy
and win more, or match common human actions and still lose every important
position.

## What would realistically have to change

If I resumed the project, I would stop another long run unless it changed the
data-generating process or the decision target.

First, I would build a proper benchmark before more training. I would freeze
`batch-005`, `batch-003`, and the pre-public Qwen policy, then run paired seeds
across many player and opponent teams. Each candidate would face the same
initial states and opponent policies. Promotion would require a confidence
interval above zero over at least several hundred paired games, plus no
regression on the public replay set. A 24-16 mirror result would be treated as a
pilot, not a championship.

Second, I would generate orders of magnitude more strategic labels. The useful
unit is not a random teacher action. For each state I would run a stronger
search teacher with several opponent-set hypotheses and save action values or
visit counts for every legal action. I would deliberately oversample positions
where switching, Tera, setup, hazard control, recovery, and sacrifice decisions
matter. The model would then learn rankings and value gaps, not only the
teacher's argmax. This is expensive, but it addresses the behavior visible in
the traces.

Third, I would give the policy an explicit belief and planning representation.
The state should track possible opponent items, abilities, moves, speed ranges,
and Tera types as distributions updated by revealed events. Candidate scoring
should include short-horizon outcome estimates: expected damage, knockout
chance, status and hazard changes, likely reply, and the value of the resulting
next state. This can still use Qwen as an encoder. It should not expect one
language-model vector to discover all of those calculations from action labels.

Fourth, I would change the objective and the deployment gate together. A new
policy must improve top-action ranking on high-impact held-out states, rare
action recall, calibration, and paired battle score. Soft KL can remain a
secondary metric. It cannot promote a model by itself. Public learning would
need a much larger replay buffer, off-policy correction, opponent and team
diversity, and enough games to estimate whether a change is real. I would keep
the public ladder as evaluation data until the local system showed a clear,
repeatable gain.

Finally, I would reconsider the 0.5B constraint only after that benchmark and
dataset existed. A controlled comparison between the same structured policy
with Qwen 0.5B, a larger encoder, and no Qwen would tell me what the language
model contributes. Right now, replacing it would mix model capacity with every
other unresolved variable.

These changes are closer to building a search-informed competitive agent than
fine-tuning a small language model. That is the realistic scope of getting past
50%. If I keep the rule that the policy must learn without a battle-search
engine at inference time, the search still has a useful role during data
generation. Refusing both inference-time search and large-scale search labels
leaves the student trying to discover long-horizon Pokémon strategy from sparse
actions and a few hundred wins. The results give me no reason to expect that to
work.

## The checkpoint I would have tested next

The best-supported checkpoint available at that point was:

```text
outputs/public-learning/positive-winrate-1000/batch-005/candidate
```

I would use it for the next test because it is the end of the accepted
direct-comparison chain. It beat `batch-003` 24-16 locally, and every later
candidate was rejected. It is not proven better on the public ladder; that
missing measurement is exactly what the next frozen run should collect.
The highest previous 100-game public set was `batch-003` at 40%, but the same
checkpoint scored 33% in its next set. I do not have evidence that those two
models are cleanly ordered on public play.

For one deterministic 100-game measurement, I would run:

```bash
python -m pokemon_battler.public_play \
  --mode ladder \
  --games 100 \
  --checkpoint outputs/public-learning/positive-winrate-1000/batch-005/candidate \
  --output-dir reports/public/batch005-frozen-100-001
```

That command does not train. It measures one frozen greedy policy, which makes
the result interpretable. A second frozen set needs a new output directory, and
I should pool the sets instead of reporting only the highest one.

If I specifically want to repeat the old learn-between-sets campaign, starting
from `batch-005`, the command is:

```bash
python -m pokemon_battler.public_play \
  --mode ladder \
  --games 100 \
  --batches 10 \
  --stop-win-rate 0.5 \
  --learn \
  --checkpoint outputs/public-learning/positive-winrate-1000/batch-005/candidate \
  --output-dir outputs/public-learning/batch005-campaign-002
```

That second command answers a different question because a promoted PPO
candidate can become the source for the next set. It reports each 100-game
record and the aggregate, but the highest set may belong to a different model.
Given the results so far, I would run the frozen command first and would not
treat “keep going until one set is positive” as evidence of improvement.

## The next run has to change the data scale

The postmortem left me with a practical question: what large offline dataset
would actually change this project? “Collect more games” is not an answer when
the collection is another small loop over the same teams.

The two Foul Play DAgger rounds contain about 51,000 turn rows from 1,000
battles. They are better labels than I first gave them credit for. Every legal
action has a search value, and the second round lets Qwen act in more than half
the visited states. The real weakness is that training used six opponent teams.
The data can be soft, on-policy, and search-backed while still covering almost
none of Gen 9 OU.

Metamon now publishes the data that this experiment was missing. `pac-base`
contains 11 million trajectories across its supported formats, and
`pac-exploratory` adds seven million higher-temperature trajectories. Its May
2026 team sets contain 139,000 Gen 9 general-ladder teams and 43,000 high-ladder
teams. I do not need to generate that coverage a hundred public games at a
time.

Using it creates a new engineering problem. Running Qwen over millions of
prepared turns would dominate the experiment before training began. Four Qwen
processes would make that worse by duplicating the model in VRAM. I therefore
split the final policy in a way that preserves the project constraint instead
of pretending the cost does not exist.

Qwen remains the accepted base policy. A second interaction head learns from
the numeric and categorical battle tensors without a language-model forward
pass. At deployment, both heads score the same 13 legal action slots and their
log probabilities are blended. This is not a move back to a heuristic battle
engine. Both distributions are learned, and Showdown still supplies rules
rather than recommendations. It is also not the old no-Qwen ablation: setting
the sidecar weight to zero recovers the exact Qwen policy.

I kept both winning and losing trajectories. The action-value head learns WIN
as one and LOSS as zero for the recorded action, the state value uses an
expectile target, and advantage-weighted cloning decides how strongly the actor
should copy it. That is more defensible than training on wins alone, which
would erase useful defense and recovery decisions from games lost later. It
also avoids treating every high-temperature exploratory action as an equally
good hard target. It still cannot say which unplayed move would have won; only
a search trace or simulator branch can supply that counterfactual.

The data path had to change with the model. The old preparation loop decoded a
20 GB archive sequentially and wrote one monolithic split. An initial version
of the larger pipeline also made the mistake of expanding the official outer
TAR.LZ4 before sampling. The corrected path reads that compressed archive
directly, rejects sampled-out members from their filenames before inner JSON
decoding, and sends selected decompression and state construction to four
spawned workers. It commits atomic JSONL shards and records a manifest after
each shard. Feature caches are built four shards at a time. Each JSONL shard has
a binary byte-offset index, and actions, outcomes, and battle lengths are cached
next to the mechanics arrays so later epochs do not parse the large JSON again.
Restarting a run reuses every completed shard; a compressed stream must be
rescanned to reach the saved boundary, but no expanded TAR is written.

I also added a parity gate before treating Metamon as compatible. Both projects
say A0 through A3 are sorted moves, A4 through A8 are sorted switches, and A9
through A12 are Tera versions of the move slots. The preparation report now
counts every recorded action that cannot be recovered from that contract. It
cannot silently turn an unmapped action into a plausible training target.

The complete run is:

```bash
python -m pokemon_battler.large_offline_pipeline \
  --output-dir outputs/metamon-large-v1
```

It originally defaulted to 5%, but an exact scan found 6,961,526 Gen 9 OU
trajectories and the first quarter of preparation already occupied 61 GiB. I
reduced the default to a deterministic 0.5%, projected to retain about 34,800
trajectories and 1.27 million decisions. Preparation now has a 32 GiB hard limit
and cache construction has a 16 GiB estimate gate. The full-data switch is
`--trajectory-sample-rate 1`; it is a storage decision, not a different model.
CPU preparation, cache generation, team parsing, and data loading use four
workers. Local evaluation runs four battles concurrently. Training still uses
one GPU and one model copy.

This is the first proposed run after the failure analysis that changes the
number and diversity of strategic states by orders of magnitude. It still may
fail. Logged self-play actions are not counterfactual outcomes, the sidecar has
only four explicit history events, and a stronger source policy can teach its
own biases. A local Foul Play win is not a public ladder result. But another
failure would now eliminate a real hypothesis: that the project was mostly
starved for broad offline experience. That is more useful than watching a fifth
loss curve settle at a slightly different number on the same data.

## The larger dataset finally changed battle performance

The first Metamon run retained about 35,000 trajectories and 1.27 million
decisions from the official self-play archives. That was a different scale from
the 1,000-battle Foul Play collection and the 100-game public PPO batches. It
also covered far more teams and positions.

I kept Qwen as the base policy and trained a structured sidecar beside it. The
base policy supplied its legal-action distribution. The sidecar scored the same
actions from the numeric mechanics, categorical identities, stable rosters,
and recent history that I had built earlier. Their legal log probabilities were
blended at deployment. Qwen still contributed its learned policy; the new head
could learn broad battle interactions without another language-model forward
pass for every training row.

The training target also used more than the recorded move. Both winning and
losing trajectories stayed in the data. An action-value head learned the final
battle outcome, a state-value head learned an expectile target, and
advantage-weighted cloning changed how strongly the actor copied each recorded
decision. The labels still could not reveal the outcome of actions that were
never played, but a loss no longer disappeared from training and a win no
longer made every action equally convincing.

The interaction cache made this practical on my hardware. It stored the
prepared mechanics tensors, identities, actions, outcomes, and row offsets in a
form the sidecar could read directly. Later epochs did not have to parse more
than a million JSON rows again, and the frozen Qwen checkpoint did not have to
be loaded during sidecar training.

On the first 100-game held-out Foul Play schedule, the Metamon v1 policy went
48-52. The old champion went 30-70 on the same schedule. An 18-point change in
the point estimate was the first local result large enough to justify extending
a policy instead of starting another architecture from scratch.

## The second policy continued the first one

The v2 run sampled the next non-overlapping 0.5% slice of the archives. It added
34,735 trajectories and 1,259,031 transitions, including 1,131,511 training
transitions. More importantly, it loaded the v1 structured head and continued
training it. Earlier experiments had sometimes restarted a candidate or
reinitialized the head I meant to improve. This run verified that continuation
before training began.

I mixed 377,170 examples from v1 back into each epoch, exactly 25% of the
combined training set. That rehearsal buffer protected the first policy's
coverage while the second hash slice supplied new positions. I lowered the
learning rate to `3e-5`, kept an epoch-zero rollback point, and trained for
three epochs. Across validation, exact action accuracy rose from 60.94% to
62.20%, switch accuracy rose from 50.36% to 52.83%, and loss fell from 1.5474
to 1.4964. The change was modest, but it moved the difficult switch slice in
the right direction without erasing v1.

I then selected the Qwen-sidecar blend with battles instead of choosing it from
the training loss. A sidecar weight of 0.75 won 30 of 50 validation games while
v1 won 24. On a separate 200-game held-out schedule, v2 finished 109-91 and v1
finished 96-104. That was a 54.5% result against 48.0%, a 6.5-point advantage.

The automatic selection pointer still stayed on v1 because its paired
bootstrap interval for the difference ran from -2.5 to 15 points. I had written
the gate to require the whole interval to clear zero. That was a sensible rule
for automatic promotion, but v2 now had the best point estimate in the blend
sweep and the larger held-out test. I chose it for the public measurement while
leaving the strict gate and its result intact.

## The final public measurement finished 502-498 over 1,000 games

I ran `outputs/metamon-large-v2/04-candidate` as a frozen greedy policy on the
public Generation 9 OU ladder. There was no PPO update and no learning between
games. Four battles ran concurrently, the submitted team stayed fixed, and the
team-preview lead was randomized.

The first trace finished 53-47 over 100 games. I later collected another 900
games with the same checkpoint and deployment settings; that set finished
449-451. The two traces share no battle IDs, so I combined them into one
1,000-game measurement against 770 different opponents. I counted every
completed game and did not select a favorable subset.

| Public result | Value |
| --- | ---: |
| Wins | 502 |
| Losses | 498 |
| Win rate | **50.2%** |
| 95% Wilson interval | 47.1%-53.3% |
| Decisions | 30,385 |
| Policy fallbacks | 0 |
| Mean battle length | 24.8 turns |

The 53% result from the first 100 games did not hold as the sample grew. The
next 900 were almost exactly even, and the final positive margin was four
games. The confidence interval still includes 50%, so this run does not prove
that the policy's underlying ladder win rate is above even. It does give me a
far better estimate than the first 100 games and clears the 35.6% aggregate
from the earlier five-batch PPO campaign.

The Showdown account `ATSskipper5` started at **1000 ELO** and showed **1189
ELO** on August 18, 2026, a net increase of 189 points. That account rating
includes earlier policies, cancelled runs, and disconnect losses as well as
these 1,000 games, so I cannot assign the whole increase to this checkpoint.
Its clean result is the 502-498 record. The account rating shows where the bot
ended up on the live ladder.

Long public collections exposed a client problem too. `poke-env` can send a
ladder search and wait forever for a battle-start notification. If Showdown
drops or expires that search without delivering the notification, the process
stays connected but stops requesting games. I stopped and resumed the run from
its append-only trace when that happened. Resume counted only completed battle
IDs, and I verified that the two source traces had no overlap. A matchmaking
watchdog that retries stale searches without touching active battles is still
needed for unattended runs.

## What finally made the difference

The positive 1,000-game result came from the accumulated representation work
and a much larger offline dataset. The mechanics and identity features solved
collisions that the old numeric summary could never separate. Qwen remained
useful as the base distribution, while the sidecar handled candidate-to-state
interactions in a compact form. Blending the two preserved behavior the small
language model had already learned and gave the structured policy enough
weight to change actual decisions.

Data coverage mattered more than another small hyperparameter change. The
earlier teacher and PPO loops recycled hundreds of games across a narrow team
pool. Metamon supplied more than a million decisions from tens of thousands of
trajectories. Training on a second disjoint slice, continuing the learned head,
and rehearsing v1 examples added coverage without throwing away the first
gain.

The evaluation process improved too. I stopped selecting checkpoints from loss
alone. The blend sweep used held-out teams, the final comparison used another
200-game schedule, and the public measurement froze the exact checkpoint under
test. Zero fallbacks across 30,385 decisions confirmed that the result came
from the policy path rather than an emergency heuristic.

This leaves `outputs/metamon-large-v2/04-candidate` as the strongest measured
policy in the project. It posted a positive 1,000-game public record without
changing the 0.5B Qwen base model or using a battle-search engine at inference
time. The margin is too small to claim a reliably positive underlying win rate,
but it is the longest and strongest public measurement I completed.
