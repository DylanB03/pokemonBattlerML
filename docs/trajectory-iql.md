# Whole-trajectory memory and next-state IQL

This is the first training path in the repository that actually preserves a
player's consecutive decisions and backs an action value up from the next
battle state. It is an extension of the existing Qwen interaction policy, not a
replacement for it.

Run the complete experiment from the repository root:

```bash
python -m pokemon_battler.trajectory_pipeline \
  --output-dir outputs/trajectory-iql-v1
```

The default starts from
`outputs/public-learning/positive-winrate-1000/batch-005/candidate`, uses the
fixed player team already bundled with the live runner, and evaluates on the
randomized nine-team Foul Play pool. It does not modify or overwrite the source
checkpoint.

## What was missing before

The schema-3 replay set sampled individual turns. Its 286,059 training rows came
from 127,229 battles, or only about 2.25 retained decisions per battle. The
interaction Transformer could attend to the current global state, twelve roster
slots, four compressed history events, and thirteen action candidates, but it
reset at every decision. It was an interaction encoder within one state, not a
trajectory model.

The old offline outcome objective also attached the final WIN or LOSS label to
every retained turn. That can say whether a trajectory eventually succeeded,
but it cannot implement a temporal-difference target because there is no paired
next state. An early good move followed by a later mistake receives the same
label as the mistake.

Schema 4 changes those two facts:

- sampling happens once per battle, so a selected POV retains all of its
  observable decisions in order;
- the two POV files from one Showdown battle get distinct `trajectory_id`
  values and can never be joined accidentally;
- unrevealed `-1` actions create a semi-Markov gap rather than breaking the
  sequence;
- every valid action stores the next valid decision, number of skipped raw
  transitions, dense transition rewards, previous observed action and reward,
  and a terminal flag;
- only the final transition receives the dominant `+1` win or `-1` loss reward.

The dense reward is intentionally small. HP change is worth at most 0.02 for a
full active-Pokémon health swing, a faint is worth 0.05, and an inflicted status
is worth 0.02. These terms help assign credit between decisions without making
damage farming more important than winning.

## Model boundary

Qwen and the existing structured interaction encoder are frozen and run once
per retained state. The cache stores:

- one 384-dimensional global battle embedding;
- thirteen 384-dimensional candidate embeddings;
- the exact legal-action mask;
- action, reward, terminal, gap, previous-action, and previous-reward fields.

The cache is a group of NumPy memory maps, not one giant `torch.load` object.
The largest array is read in short trajectory windows, so training does not
need to copy the full cache into RAM. Qwen remains part of every deployed
decision; caching only avoids repeating its frozen forward pass during head
training.

Two heads are trained from exactly the same cached rows:

1. `memoryless` uses the current global and candidate embeddings plus the
   previous action/reward fields, but has no persistent hidden state.
2. `recurrent` adds a two-layer GRU. It carries learned hidden state across all
   decisions in a battle and uses a 16-turn burn-in when a long trajectory is
   split into 64-turn windows.

Both heads contain a legal-masked 13-way actor, two candidate Q functions, and
a state-value function. The memoryless run is a required control: if the GRU
does not win more games on the same paired schedule, the pipeline does not
select it merely because it is newer.

## Objective

For logged action `a_t`, the Q target is:

```text
r_t + gamma^gap * (1 - done_t) * V_target(s_next)
```

The two Q heads regress that target. The value head uses an expectile loss
against the smaller Q estimate. The actor starts with one behavior-cloning
epoch, then uses advantage-weighted regression while retaining a small cloning
term. A slowly updated target network supplies the next-state value.

This is offline IQL-style training. It never asks the offline critic to score an
unseen action as a supervised target, and it does not pretend the final battle
label is a per-turn Q value.

## Live inference

A trajectory checkpoint contains the original Qwen adapter and interaction
head plus `trajectory_head.safetensors`. Existing local and public runners
detect that file automatically.

The live player keeps one GRU hidden state per battle tag. This matters when
several games run concurrently. It also caches the last prediction for a
Showdown request, so a resent request neither samples a second action nor
advances the hidden state twice. Hidden state is deleted when the battle ends.

Frozen local and public play work through the existing commands. The older
between-batch PPO updater is deliberately rejected for these checkpoints: it
shuffles turns and updates the statewise interaction policy, so applying it to
actions sampled by the GRU would be mathematically wrong. Ordered temporal PPO
needs its own rollout updater before `public_play --learn` can be used with a
selected trajectory checkpoint.

## End-to-end stages and artifacts

The run directory is append-free and self-contained:

```text
01-trajectories/                 schema-4 train/validation/test JSONL and report
02-encoded-cache/train/          frozen Qwen+interaction memory maps
02-encoded-cache/validation/
03-memoryless/                   complete deployable control checkpoint
04-recurrent/                    complete deployable GRU checkpoint
05-recurrent-vs-memoryless/      paired 100-game architecture comparison
06-selected-vs-champion/         paired 100-game comparison with the source model
selected_checkpoint.txt
manifest.json
```

The first battle suite chooses between recurrent and memoryless from complete
games. The second compares that choice with the source champion on a new seeded
schedule. Every checkpoint and rejected result remains on disk.

For a preparation/training smoke test without starting Showdown, use a new
output directory and add:

```bash
--max-trajectories-per-split 32 --epochs 1 --skip-battle-evaluation
```

That flag is for wiring only. Thirty-two trajectories cannot establish a
policy improvement.

## What this does not prove

Offline validation still measures recorded-action agreement and critic losses;
neither is the project goal. The paired Foul Play suites are the selection
criterion, and public ladder play remains the final external test. A 100-game
result has substantial uncertainty, so the report includes a paired bootstrap
interval and retains the old champion even when a candidate is selected.

This change fixes a concrete representation and credit-assignment defect. It
does not guarantee 51% ladder win rate, turn Foul Play into part of inference,
or increase Qwen beyond 0.5B parameters.
