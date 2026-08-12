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

## What is collected

Every local evaluation against `foul-play` now also writes
`foul_play_teacher.jsonl` in its report directory. Each decision contains:

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

Run this after the current public campaign is finished. It launches the pinned
Foul Play checkout on the local Showdown server and does not affect public ELO:

```bash
python -m pokemon_battler.live_eval \
  --checkpoint outputs/your-current-checkpoint \
  --opponent foul-play \
  --games 100 \
  --foul-play-search-time-ms 250 \
  --output-dir reports/teacher/foul-play-001
```

The normal result is saved to `summary.json`; `teacher_examples` reports how many
usable decisions were collected. More search time generally produces a better
teacher target but makes collection proportionally slower. The old 100 ms value
is appropriate for a quick integration check. Prefer 250-500 ms for a dataset
that will actually train a checkpoint.

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
