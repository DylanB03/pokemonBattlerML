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

## Fixed-team perspective

The teacher must make decisions from the model's deployment perspective. Foul
Play therefore controls the model's one fixed team during collection. Its
opponent receives a different team from a shuffled OU pool before every battle.
Randomizing Foul Play's own team would be backwards: the resulting examples
would teach Qwen as though its own roster changed while the enemy stayed fixed.

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
- Foul Play's selected action, confidence, entropy, and total search visits;
- the raw Foul Play choice distribution for auditing.

The collector taps the distribution before Foul Play discards choices below 75%
of its best choice. That softer target preserves information such as “switching
is plausible, but this attack is somewhat better” instead of reducing every
position to one noisy hard label.

The saved observation contains only what Foul Play knew before it sampled
opponent sets for search. Sampled hidden items, moves, abilities, and spreads are
not copied into the student's input.

## Collect teacher games

First save multiple valid Gen 9 OU Showdown exports in an enemy-team directory.
Use different compositions and archetypes, not reordered copies of one team.
Smogon's maintained [SV OU sample teams](https://www.smogon.com/forums/threads/sv-ou-sample-teams.3712513/)
are one suitable source.

Then run this after the current public campaign is finished. It launches the
pinned Foul Play checkout on the local Showdown server, keeps the model team
fixed, and does not affect public ELO or load Qwen:

```bash
python -m pokemon_battler.teacher_collect \
  --team-file examples/teams/gen9ou-balance.txt \
  --enemy-team-dir data/teams/gen9ou-enemies \
  --games 100 \
  --foul-play-search-time-ms 250 \
  --output-dir reports/teacher/foul-play-001
```

`summary.json` explicitly records `teacher_team_fixed: true` and
`enemy_teams_randomized: true`. `enemy_team_selections.json` records the exact
team used in each battle, its result, the selection order, and per-team counts,
while `teacher_examples` reports how many usable decisions were collected. More
search time generally produces a better teacher target but makes collection
proportionally slower. Prefer 250-500 ms for a dataset that will actually train
a checkpoint.

Individual enemy files can be supplied instead of a directory by repeating
`--enemy-team-file`. Collection parses the exports and fails before starting
Showdown if fewer than two distinct species compositions resolve; reordered
copies of one team do not count as diversity.

Multiple runs can be joined before training:

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

The loss is computed only over exactly legal candidates. It uses the soft MCTS
distribution, retains a small hard-label term for Foul Play's selected choice,
and gives extra weight to states where a confident teacher disagrees with the
student. Weight normalization keeps the effective learning-rate scale stable.

Use `--validation-data` with a teacher trace from different battles when enough
data has been collected. Without it, the before/after measurements use the
training file and measure fit rather than generalization.

## What this does and does not prove

This first stage is policy distillation on Foul Play's visited states. It gives
Qwen supervised examples of search-backed switches, setup moves, recovery,
hazards, Tera decisions, and attacks. The deployed model remains Qwen plus the
interaction head and does not call Foul Play or `poke-engine` at battle time.

It is not yet DAgger: Foul Play is not shadow-searching every state visited by a
Qwen-controlled player. If offline teacher agreement rises but battle win rate
does not, the next useful extension is a shadow-advisor pass over student-visited
states, followed by another distillation round. Promotion should still be based
on held-out games against Foul Play, the heuristic opponents, and the previous
checkpoint—not training loss alone.
