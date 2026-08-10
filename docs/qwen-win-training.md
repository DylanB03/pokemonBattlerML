# Training Qwen for battle wins

The win-training pipeline changes the optimization target without changing who
makes the decisions. Qwen and the interaction transformer assign probabilities
to the 13 battle actions. The exact Showdown request removes illegal actions.
Pokémon Showdown applies the selected action and returns the next public
observation and final result. There is no search, MCTS, damage planner,
`poke-engine`, or external policy in this loop.

The runnable pilot is:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-pilot-1
```

Use a new output directory for every run. The command refuses to write into a
nonempty directory and never overwrites the behavior-cloning checkpoint or an
earlier candidate.

## What the command does

It first reuses `data/gen9ou-interaction-v1/train.jsonl` and its interaction
cache. If the prepared dataset is absent, it prepares the same chronological
schema-3 split from `data/raw/metamon/gen9ou.tar.gz` and builds the cache. Each
row already retains its battle ID, turn, legal mask, recorded action, final
WIN/LOSS result, visible roster, and strictly backward-looking history.

The first optimization phase loads
`outputs/interaction-v1-1epoch/policy/final` as a warm start. It adds a separate
win-value estimate for every legal action and fits three related quantities:

- `Q(s,a)`, the selected action's probability of eventually winning;
- `V(s)`, the state's expectile win value;
- `pi(a|s)`, Qwen's legal-action policy, weighted toward positive estimated
  advantages.

The archive has terminal outcomes for complete battles but no counterfactual
next state for actions that were not chosen. The offline Q target is therefore
the logged action's Monte Carlo battle result. This is an IQL-style
outcome-conditioned warm start, not a claim that the data contains outcomes for
all 13 actions. A small ordinary behavior-cloning term prevents the policy from
moving sharply while the new value heads are learning.

The second phase repeatedly performs:

1. Sample a frozen Qwen checkpoint from the league.
2. Sample legal actions from the current Qwen policy in complete local
   Showdown games.
3. Store the public observation, action, exact old log probability, old value,
   terminal reward, GAE advantage, and return for every actor decision.
4. Run clipped PPO updates on those saved decisions.
5. Play the candidate deterministically against the current champion.
6. Promote it only when its score meets the configured threshold.

The only reward is `+1` for a win, `-1` for a loss, and `0` for a tie. There is
no HP, knockout, damage, or move-quality shaping that could reward something
other than winning. PPO uses the exact legal-action distribution, a clipped
value update, entropy regularization, gradient clipping, and early stopping
when the approximate KL moves too far from the rollout policy.

## Default pilot budget

The default is deliberately a pilot that can expose whether win rate moves
before committing to a much larger run:

```text
offline updates          1,000
self-play iterations     4
rollout games/iteration  64
PPO epochs/rollout       3
promotion games          40
promotion score          55%
```

These counts are not enough to establish a final ladder strength. Forty
promotion games have a wide confidence interval. Once the curve is healthy,
increase `--rollout-games` and `--promotion-games`; the command reports a
Wilson interval so the uncertainty remains visible.

The default team pool rotates the bundled valid team's order so every member is
used as a lead. For genuine team-distribution training, pass several team files:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-team-pool-1 \
  --team-file examples/teams/gen9ou-balance.txt \
  --team-file /path/to/another-valid-gen9ou-team.txt \
  --team-file /path/to/a-third-valid-gen9ou-team.txt
```

The current PPO action space begins after team preview. The default rotations
remove the old always-lead-Great-Tusk shortcut, but they are not a learned
team-preview policy. A learned preview head needs preview decisions in the
rollout schema and should be judged by the same terminal reward.

## Artifacts and selection

Every model remains available:

```text
outputs/qwen-win-pilot-1/
  offline/                         outcome-trained warm start
  league.json                      frozen population and match history
  team-pool/                       six deterministic lead rotations
  iteration-001/
    rollout/rollouts.jsonl         complete PPO training rows
    rollout/summary.json
    candidate/                     post-PPO Qwen checkpoint
    promotion/summary.json
    iteration_summary.json
  selected_checkpoint.txt          current promoted champion
  summary.json
```

A rejected candidate is not deleted. It remains in its iteration directory but
does not enter the opponent population. The original behavior-cloning policy is
retained as a frozen regression opponent. Promoted checkpoints are also frozen
and sampled in later iterations, including older policies, which reduces
latest-versus-latest co-adaptation.

## Useful controls

Run only the self-play/PPO portion from the existing checkpoint:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-no-offline-1 \
  --skip-offline
```

Run a small wiring test:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-smoke-1 \
  --offline-max-steps 1 \
  --iterations 1 \
  --rollout-games 2 \
  --promotion-games 2 \
  --concurrent-games 1
```

Scale a later run:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-main-1 \
  --offline-max-steps 4000 \
  --iterations 12 \
  --rollout-games 256 \
  --promotion-games 200
```

The default uses 4-bit QLoRA and local model files. `--allow-download` permits a
missing Qwen checkpoint to be downloaded. `--no-4bit` is intended for a machine
with enough memory and is also useful for a CPU wiring test, though CPU training
is not a practical main run.

## What to watch

Offline loss is only a wiring and representation diagnostic. During PPO, watch
approximate KL, clipping fraction, entropy, value loss, and the promotion match.
The primary result is the selected checkpoint's win rate across frozen Qwen
opponents and teams. Human-action accuracy remains useful for detecting a
catastrophic loss of basic play, but it no longer selects the model.

The system currently loads two 0.5B Qwen policies for a self-play match and one
trainable policy for an update. Battles can run concurrently, but synchronous
policy calls are not yet combined into one GPU batch. That batching optimization
can improve throughput without changing the learning objective.

Completed public games can now feed the same rollout schema through the
separate `pokemon_battler.public_play` runner. Public PPO is opt-in, runs only
between frozen game batches, uses smaller default learning rates, and retains
the existing local champion-promotion gate. Account setup and commands are in
[Public Showdown play and between-game learning](public-showdown-learning.md).
