# Foul Play policy distillation

The project can now use Foul Play as an offline teacher without putting
`poke-engine` in the deployed policy. Foul Play searches local games with MCTS,
the harness records its complete visit distribution, and a separate training
command teaches the Qwen interaction policy to match it.

## A0-A12 are internal candidate slots

Qwen does not generate the strings `A0` through `A12`. The runtime renders the
battle state, takes Qwen's final hidden state, combines it with the structured
interaction transformer, and scores a fixed table:

- `A0-A3`: the active Pokémon's four moves, sorted by normalized move name;
- `A4-A8`: up to five bench Pokémon, sorted by normalized species name;
- `A9-A12`: the same four moves with Terastallization.

Showdown's current request supplies the exact legal mask. An unavailable move,
trapped switch, or unavailable Tera candidate is assigned zero probability. The
stable indices are necessary to align a Foul Play choice such as `stealthrock`,
`switch corviknight`, or `thunderbolt-tera` with the same structured candidate
that the interaction head scores. They are labels in the data, not language the
model must learn to emit.

## Strong-vs-strong fixed-team perspective

The teacher must make decisions from the model's deployment perspective. One
Foul Play instance therefore controls the model's fixed team and records its
full MCTS policy. A second, independent Foul Play instance controls the opponent
and receives a different team from a shuffled OU pool before every battle. Both
use the same search budget by default. Randomizing the teacher's own team would
be backwards: the resulting examples would teach Qwen as though its deployment
roster changed.

This distinction matters. A strong teacher playing a weak heuristic still
produces strong labels, but the weak opponent determines which states occur. A
42-1 teacher record, for example, mostly samples responses to exploitable play
and underrepresents difficult switches, preservation, setup, prediction, and
endgames. Foul-Play-vs-Foul-Play makes both the action targets and the state
distribution search-backed. `--enemy-policy heuristic`, `max-power`, and
`random` remain available only for controlled ablations and wiring tests.

The dedicated collection command requires at least two distinct enemy team
files. It samples without replacement within shuffled cycles and prevents the
same team from appearing on both sides of a cycle boundary. The fixed team file
is never placed in that pool implicitly.

## What is collected

`pokemon_battler.teacher_collect` writes `foul_play_teacher.jsonl` in its output
directory. Each decision contains:

- the public battle observation from Foul Play's side;
- player and opponent rosters plus recent transitions;
- the exact legal candidate mask;
- the unfiltered MCTS visit distribution across all mapped legal choices;
- each searched action's average MCTS win value and the root value;
- Foul Play's selected action, confidence, entropy, and total search visits;
- the final battle outcome, enemy team, and number of decisions in the trajectory;
- the raw Foul Play choice distribution for auditing.

There is also one `decision_phase: team_preview` row per battle. It contains the
six-way soft lead policy and action values used to train the separate lead head.
At runtime that head evaluates the fixed deployment team against the opponent's
public preview; the bot no longer rotates or hard-codes slot one when the head is
present.

The collector taps the distribution before Foul Play discards choices below 75%
of its best choice. That softer target preserves information such as “switching
is plausible, but this attack is somewhat better” instead of reducing every
position to one noisy hard label.

The saved observation contains only what Foul Play knew before it sampled
opponent sets for search. A shared canonicalizer removes unrevealed HP, item,
ability, Tera, status, and move fields and reconstructs the same turn/history
context used by live inference. Sampled hidden sets are never copied into the
student's input.

## Collect teacher games

First save multiple valid Gen 9 OU Showdown exports in an enemy-team directory.
Use different compositions and archetypes, not reordered copies of one team.
Smogon's maintained [SV OU sample teams](https://www.smogon.com/forums/threads/sv-ou-sample-teams.3712513/)
are one suitable source.

Then run this after the current public campaign is finished. It launches the
pinned Foul Play checkout twice on the local Showdown server, keeps the model
team fixed, and does not affect public ELO or load Qwen:

```bash
python -m pokemon_battler.teacher_collect \
  --team-file examples/teams/gen9ou-balance.txt \
  --enemy-team-dir data/teams/gen9ou-enemies \
  --games 100 \
  --foul-play-search-time-ms 250 \
  --output-dir reports/teacher/foul-play-001
```

The default `--enemy-policy foul-play` means the matchup is strong-vs-strong.
`PBFoulPlay` is the fixed-team teacher and `PBFoulPlayEnemy` is the randomized
opponent. These are temporary users on the local `--no-security` server; the
collector does not read `.env`, connect to public Showdown, or touch ladder ELO.
The terminal prints `[teacher N/100]` with the current record after every game.
Detailed search logs are saved to `opponent.log` and `enemy/opponent.log`.

Before launching either bot, the collector validates the fixed team and every
enemy team against the installed Showdown rules. An obsolete team now fails
immediately with its exact legality error instead of leaving challenge handling
waiting forever for a battle that Showdown rejected. A ten-minute no-log-progress
watchdog provides a second failure boundary and can be adjusted with
`--battle-stall-timeout`.

`summary.json` explicitly records `teacher_team_fixed: true`, both Foul Play
revisions and search budgets, and `enemy_teams_randomized: true`.
`enemy_team_selections.json` records the exact enemy team used in each battle,
its result, the selection order, and per-team counts, while `teacher_examples`
reports how many usable decisions were collected. More search time generally
produces a better teacher target but makes collection proportionally slower.
Prefer 250-500 ms for a dataset that will actually train a checkpoint. Use
`--enemy-foul-play-search-time-ms` only when deliberately testing an asymmetric
opponent; otherwise it inherits the teacher's value.

Individual enemy files can be supplied instead of a directory by repeating
`--enemy-team-file`. Collection parses the exports and fails before starting
Showdown if fewer than two distinct species compositions resolve; reordered
copies of one team do not count as diversity.

Multiple runs may be joined before training, but only combine collections made
with the same strong-vs-strong setup. The end-to-end pipeline does this itself
and never imports the old heuristic-opponent traces.

```bash
find reports/teacher -name foul_play_teacher.jsonl -print0 \
  | xargs -0 cat > data/foul_play_teacher.jsonl
```

## Distill into a new checkpoint

```bash
python -m pokemon_battler.distill \
  --checkpoint outputs/your-current-checkpoint \
  --teacher-data data/foul_play_teacher.jsonl \
  --output-dir outputs/foul-play-distilled-001 \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

The source checkpoint is never overwritten. The output is a complete new policy
checkpoint with `distillation_report.json`. The terminal prints measurements
before and after the update:

- `teacher_top1_agreement`: how often student and teacher rank the same action
  first;
- `teacher_student_kl`: distance from the complete teacher distribution;
- `soft_cross_entropy`: the optimization target before example weighting;
- `teacher_confidence` and `student_entropy`: useful checks for noisy or collapsed
  policies.

The loss is computed only over exactly legal candidates. Its default is now the
complete soft MCTS policy—there is no random hard-choice term and no confidence
weight that suppresses uncertain Tera examples. It also trains soft action-family
targets, per-action MCTS values, and the root value. Tera-bearing policies receive
controlled extra weight, while each battle has a capped total contribution so a
single 200-turn game cannot dominate dozens of normal games. The deployed scorer
uses the learned action value as a conservative secondary score.

Qwen is frozen by default during this small teacher update. The interaction head
does the first adaptation, and `--rehearsal-data` mixes human replay decisions to
limit forgetting. Pass `--no-freeze-qwen` only after a larger teacher corpus has
shown held-out gains. When no explicit validation file is given, the trainer now
holds out complete enemy-team compositions when metadata permits, otherwise
complete battles. It never reports the training set as validation and restores
the best validation head before saving.

## What this does and does not prove

The deployed model remains Qwen plus learned PyTorch heads and does not call Foul
Play or `poke-engine` at battle time. Collection can now run DAgger by giving the
student control of a fraction of the fixed-team teacher's decisions while Foul
Play searches and labels every state it visits:

```bash
python -m pokemon_battler.teacher_collect \
  --team-file examples/teams/gen9ou-balance.txt \
  --enemy-team-dir data/teams/gen9ou-train \
  --student-checkpoint outputs/current-champion \
  --student-action-probability 0.7 \
  --games 200 \
  --output-dir reports/teacher/dagger-001
```

This is not ordinary teacher self-play: the `behavior.source` field proves
whether the teacher or student actually chose each action. The labels always
remain Foul Play's full search targets. Later rounds aggregate all new traces so
they expand the visited-state distribution instead of repeatedly fitting one
small batch.

## One end-to-end command

The recommended command keeps the deployment team fixed, reserves three enemy
compositions that never enter training, performs an expert round followed by
student-controlled DAgger rounds, trains the corrected turn and preview heads,
and compares every candidate with the current champion on the identical held-out
Foul Play schedule:

```bash
python -m pokemon_battler.win_pipeline \
  --checkpoint outputs/public-learning/positive-winrate-1000/batch-005/candidate \
  --team-file examples/teams/gen9ou-balance.txt \
  --enemy-team-manifest examples/opponent-pools/gen9ou-foul-play.txt \
  --output-dir outputs/qwen-dagger-v1 \
  --rounds 3 \
  --games-per-round 200 \
  --evaluation-games 100 \
  --concurrent-games 4 \
  --search-time-ms 250
```

Every round is retained under `round-NN/`; rejected candidates are not deleted.
`manifest.json` records the permanent train/held-out team split, aggregate data
size, student-control probability, promotion result, and current champion. A
candidate is promoted only when its held-out win-rate delta is positive and the
paired bootstrap interval is not strongly negative. PPO is deliberately absent
from this pipeline. It should only be reconsidered after DAgger stops improving
the held-out suite, because sparse public-game PPO was the least informative and
most failure-prone update in the previous process.

`--concurrent-games 4` runs four independent battles at once. Teacher collection
uses four isolated Foul Play pairs and merges their traces with worker-prefixed
battle IDs. DAgger workers share one loaded Qwen advisor rather than allocating
four model copies. Held-out evaluation similarly runs four Foul Play opponents
against one Qwen player, then aligns candidate and champion results by worker,
game index, and scheduled team. Keep Foul Play search parallelism and search
threads at one when using battle-level concurrency; increasing both forms of
parallelism will oversubscribe a 16-thread CPU.
