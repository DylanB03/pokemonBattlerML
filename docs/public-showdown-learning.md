# Public Showdown play and between-game learning

The public runner connects the existing Qwen interaction policy to the main
Pokémon Showdown server. It can verify an account, accept a fixed opponent's
challenges, challenge a fixed opponent, or explicitly enter the ladder. Every
battle saves the same exact legal-action observations used by local self-play.

Public learning is deliberately between games, not during a game. One frozen
checkpoint completes an entire batch. The runner then disconnects it, trains a
new candidate from that batch, evaluates candidate versus champion on the local
Showdown server, and promotes only a candidate whose score reaches the gate.
Every candidate is retained even when rejected.

## Configure the account

Copy the committed template:

```bash
cp .env.example .env
```

Fill in the registered bot account:

```dotenv
POKEMON_SHOWDOWN_USERNAME=YourClearlyNamedBot
POKEMON_SHOWDOWN_PASSWORD="replace-me"

# Optional; --opponent overrides this value.
POKEMON_SHOWDOWN_OPPONENT=YourTestingAccount
```

`.env` and every `.env.*` variant except `.env.example` are ignored by Git. The
password is used only to create `poke-env`'s account configuration. It is never
written to `run_config.json`, summaries, traces, or model metadata. Process
environment variables with the same names override the file when present.

Verify the account without loading Qwen:

```bash
python -m pokemon_battler.public_play --mode login
```

## Play frozen-policy games first

The first real games use unrated challenges from one named testing
account. Start the bot with:

```bash
python -m pokemon_battler.public_play \
  --mode accept \
  --opponent YourTestingAccount \
  --games 20
```

Then send Generation 9 OU challenges from that account in the browser. The
runner accepts only that normalized username and exits after exactly 20
completed games. To make the bot initiate the challenges instead, use
`--mode challenge`.

Frozen evaluation is deterministic at each ordinary decision. Public team
preview is randomized because the current 13-action policy begins after team
preview and should not expose a permanent slot-one lead. Use
`--team-preview first` only for a deliberately fixed-lead measurement.

`--concurrent-games` controls the maximum number of public battles in flight
for one logged-in player. Multiple frozen `--batches` are supported so a long
evaluation can retain separate per-batch records without enabling PPO. For
example, this runs five frozen 100-game suites with up to four simultaneous
battles:

```bash
python -m pokemon_battler.public_play \
  --mode ladder \
  --checkpoint outputs/metamon-large-v2/04-candidate \
  --games 100 \
  --batches 5 \
  --concurrent-games 4 \
  --team-preview random \
  --output-dir reports/public/metamon-large-v2-5x100
```

This command deliberately omits `--learn`: the structured sidecar has its own
training objective and is rejected by the older statewise PPO updater. Each
batch and the aggregate campaign still receive win/loss and ELO summaries.

If a frozen campaign is interrupted between battles, rerun the identical
command with `--resume`. The runner rebuilds the current batch from its
append-only `decisions.jsonl`, requests only the unfinished game count, and
then continues later batches. It rejects checkpoint, account, batch-size,
format, team, or preview-policy mismatches instead of combining different
experiments. For the campaign above:

```bash
python -m pokemon_battler.public_play \
  --mode ladder \
  --checkpoint outputs/metamon-large-v2/04-candidate \
  --games 100 \
  --batches 5 \
  --concurrent-games 4 \
  --team-preview random \
  --output-dir reports/public/metamon-large-v2-5x100 \
  --resume
```

I completed the first 100-game batch with the frozen Metamon v2 checkpoint. It
finished **53-47** against 88 public opponents, made 2,852 policy decisions,
and used no fallback actions. The `ATSskipper5` account showed **1152 ELO** when
I checked it afterward. That account snapshot included two later disconnect
losses, so the exact rating at the end of game 100 was somewhat higher.

Completed battle records, decisions, replay files, fallbacks, inference
latencies, and captured ELO transitions remain in the same batch summary.
Resume is intentionally limited to frozen campaigns because PPO requires one
complete on-policy rollout batch.

There is no wall-clock deadline on a battle session. A finite `--games` run
waits until every requested game has ended before closing the connection, even
when the run takes several hours. `--login-timeout` applies only while initially
authenticating, before matchmaking begins; it can never terminate a battle.

Every completed game immediately prints its result, opponent, turn count,
cumulative win-loss-tie record, win rate, and the exact old-to-new ELO change
when Showdown marks the game as rated. The end of each public batch prints the
same compact record plus an ELO line with the starting rating, ending rating,
net change from the bot's games, total points gained, total points lost, peak,
minimum, and number of captured rated games. Learning runs also announce the
start and completion of PPO training, its update count, approximate KL and
duration, followed by the candidate-versus-champion result and promotion
decision. The full machine-readable details remain in the JSON artifacts
rather than flooding the terminal.

The default checkpoint is read from:

```text
outputs/qwen-win-pilot-1/selected_checkpoint.txt
```

`--checkpoint` can point either to another checkpoint directory or another
`selected_checkpoint.txt` file.

## Learn between public batches

After the frozen run is sound, an opt-in learning session is:

```bash
python -m pokemon_battler.public_play \
  --mode accept \
  --opponent YourTestingAccount \
  --games 32 \
  --batches 3 \
  --learn \
  --output-dir outputs/public-learning/challenge-1
```

For each batch, the command:

1. loads the current frozen champion;
2. samples actions from its exact legal distribution at temperature 1;
3. plays 32 complete public games and records terminal-only win rewards;
4. refuses training if any game is incomplete, any decision fell back, or any
   trajectory remains pending;
5. saves a separate PPO candidate using conservative public-update learning
   rates;
6. plays 40 local candidate-versus-champion promotion games; and
7. promotes at a 55% score, otherwise retaining the prior champion.

The bot reconnects only after training and promotion finish. For challenge
mode, the testing opponent therefore sends the next batch after the runner is
back online. Fewer than 16 games is supported as a wiring test but emits a
warning because one or two outcomes make an extremely noisy gradient.

PPO currently requires sampling temperature `1.0`. This is enforced because
the saved old action probability must describe the exact distribution that the
candidate recomputes. Evaluation without `--learn` remains greedy by default.

### Bounded continuous campaign

This command plays consecutive 100-game learning batches, stops as soon as a
batch has more wins than losses, and otherwise stops after 1,000 public games:

```bash
python -m pokemon_battler.public_play \
  --mode ladder \
  --games 100 \
  --batches 10 \
  --stop-win-rate 0.5 \
  --learn \
  --checkpoint outputs/public-learning/ladder-001/selected_checkpoint.txt \
  --output-dir outputs/public-learning/positive-winrate-1000
```

The threshold comparison is strict: a 50-50 batch continues, while 51-49
stops. Ties count as half a point, so the equivalent rule is that wins must
exceed losses. When a batch misses the target, PPO initializes from the current
selected checkpoint. A promoted candidate becomes the next batch's source; a
rejected candidate remains on disk but is not allowed to degrade the chain.
Checkpoint directories are separate model versions, not independent training
restarts.

When a batch reaches the target, training stops before another PPO update. This
preserves the exact checkpoint whose public batch met the criterion. At the
1,000-game ceiling, the final batch is still used for PPO and local promotion;
the summary explicitly says whether the final selected candidate has itself
played a subsequent public batch.

Each `batch-NNN/batch_summary.json` contains that batch's public result, PPO
metrics, source and candidate checkpoints, promotion match, and selection
decision. `campaign_summary.json` is rewritten after every completed batch and
aggregates the public record, exact tracked ELO changes, fallbacks, candidate
counts, PPO updates, promotion record, public score change, and complete
promoted-checkpoint chain. Each item in `batch_results` includes its own rating
summary, so a 100-game suite can be compared directly by record, win rate, and
ELO outcome. `summary.json` embeds both the individual reports and the final
campaign summary.

The rating summary is based on Showdown's authoritative result line for every
rated battle (`old rating -> new rating`), not an estimated K-factor and not
poke-env's pre-battle `battle.rating` value. `net_change` sums only changes from
captured bot games. `start_to_end_change` is also saved; `untracked_change`
exposes any difference between those values, such as ladder movement from games
played on the account outside the campaign. Unrated challenge suites still
report their win-loss record and win rate, with ELO marked unavailable.

## Artifacts

A learning run stores:

```text
outputs/public-learning/challenge-1/
  run_config.json
  league.json
  selected_checkpoint.txt
  summary.json
  batch-001/
    public/
      decisions.jsonl
      rollouts.jsonl
      public_summary.json
      replays/
    candidate/
    promotion/
    batch_summary.json
```

The trace contains opponent usernames because that identity is part of the
public battle session. Remove or hash it before sharing the trace outside the
project. Model training uses observations, actions, old policy/value outputs,
and terminal results; the username is not a model feature.

## Ladder mode

The technical ladder command is explicit:

```bash
python -m pokemon_battler.public_play --mode ladder --games 20
```

Use direct challenges first. Public ladder automation is subject to Pokémon
Showdown's current rules, rate limits, bot limiters, and staff discretion. Do
not use multiple controlled accounts for rated games, manipulate usage, enter
suspect tests, timer-stall, or run an unattended high-volume loop. The runner
starts Showdown's battle timer and caps every invocation at the requested game
count. It does not abandon an active battle when an overall runtime is reached.

## What this does not learn

The action space still starts after team preview, so PPO cannot learn which
lead to select. Randomizing preview prevents a fixed shortcut but is not a
learned preview policy. Public outcome batches are also far smaller and noisier
than the replay and local self-play sets. The promotion gate protects the
current champion from a directly weaker candidate; it does not prove broader
human-ladder improvement. Keep frozen public evaluation sessions separate from
learning sessions when reporting win rate.
