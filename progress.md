# initial pre training

```
{"id":"cmpl-9523f9b435e598f7","object":"text_completion","created":1784847078,"model":"Qwen/Qwen2.5-0.5B","choices":[{"index":0,

"text":" \n\n<legal_actions_counts>\n<A0> 0\n<A1> 0\n<A2> 0\n<A3> 0\n<A4> 0\n<A5> 0\n</legal_actions_counts>",

"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-0.25.1-6cda6dc1","usage":{"prompt_tokens":326,"total_tokens":374,"completion_tokens":48,"prompt_tokens_details":null},"kv_transfer_params":null,"metrics":null}
```

# SFT harness

The first training harness is ready:

- Streams Metamon JSON, JSON-LZ4, and tar archives into grouped JSONL splits.
- Preserves Metamon's alphabetical move/switch action mapping.
- Drops missing and unrecoverably illegal replay actions.
- Renders prompts dynamically and applies loss only to the action target.
- Supports full SFT, LoRA, and 4-bit QLoRA.
- Evaluates by ranking only currently legal actions.
- Supports directly legal-masked fixed and shared candidate action heads.
- Uses a compact prompt with a measured 1,157-token median on 25k examples.
- Selects checkpoints using deterministic validation action accuracy and reports
  action NLL, top-k accuracy, MRR, family slices, entropy, and prediction counts.
- Keeps short-run limits separate from the LR schedule horizon and reports actual
  dataset passes, gradients, throughput, and clipping.
- Provides a one-command train/evaluate/report runner.
- Reports reproducible static, frequency, ranking, and action-type baselines.
- Includes a compact feature-hashed non-language-model policy baseline.
- Includes a hybrid policy with a versioned 207-value mechanics tensor and 32
  learned categorical identity fields for each action.
- Precomputes mechanics into memory-mapped float16 caches and provides a
  mechanics-only MLP ablation.
- Stops one-command runs after four flat validation checks while preserving the
  exact best checkpoint.

Verified locally with:

- 46 unit tests.
- One real Qwen2.5-0.5B LoRA optimization and adapter-save step.
- Adapter reload and constrained action evaluation.
- One real Qwen2.5-0.5B policy-head optimization, save, reload, and prediction.
- One real Qwen2.5-0.5B candidate-head optimization plus validation, best/final
  save, reload, automatic prompt/head detection, and constrained prediction.
- Non-language-model train, save, reload, and batched evaluation smoke tests.
- Mechanics feature, cache, hybrid-score, and mechanics-only backpropagation
  tests.
- One real Qwen2.5-0.5B mechanics-v2 forward/backward pass with `[1, 13, 207]`
  numeric features and `[1, 13, 32]` categorical identities.

## Mechanics-v2 representation update

The first mechanics design collapsed distinct legal actions into identical
vectors in 2,880 of 286,059 training rows. The target itself was ambiguous in
780 rows. The replacement keeps the numeric path but adds typed side
conditions, speed/order context, corrected damage cases, candidate defensive
matchups, and learned identities for moves, species, items, abilities, types,
statuses, effects, field state, and history. Compact move names remain in the
prompt as a residual signal for mechanics that should not be reduced to one
generic flag. V1 checkpoints and caches are still loadable.

## Candidate-head reference result

The compact candidate-head run was stopped after step 5,000. Its best fixed
1,024-row validation result was 374/1,024, or 36.5234% net top-1 action
agreement, with candidate NLL 1.6630. Training loss continued to fluctuate in
roughly the 1.60-1.75 band. Because 5,000 updates cover about 56% of the
286,059-row split at effective batch 32, this is neither a completed epoch nor a
5,000-row final evaluation. It is the reference the next representation must
beat.

Next experiment:

```bash
.venv/bin/python -m pokemon_battler.experiment \
  --output-dir outputs/mechanics-v2-1epoch
```

This builds or reuses the mechanics caches, trains for at most one complete
pass, stops a validation plateau automatically, and compares the best and final
checkpoints on the same validation rows.

## Mechanics-v2 completed result

The run stopped at step 8,000 after 20 hours 48 minutes. The selected final
checkpoint reached 42.86% exact action agreement on 5,000 validation rows,
64.78% top-2, 78.78% top-3, 0.6265 mean reciprocal rank, and 1.5452 candidate
NLL. On the same fixed 1,024-row validation subset used by the earlier
candidate-head run, the best mechanics-v2 checkpoint reached 41.8945% instead
of 36.5234%, a gain of 5.37 percentage points with the same 0.5B base model.

The result is uneven by action family: 50.78% ordinary-move accuracy, 31.16%
switch accuracy, and 3.57% Tera-move accuracy. The model predicted Tera only
five times for 112 Tera targets, although four of those five predictions were
correct.

A later audit found that the current `data/gen9ou-dev` split is entirely from
the legacy preparation schema. None of its train, validation, or test rows
contains recent move history, accumulated opponent reveals, or legal-mask
quality. The current preparer already writes those fields, so the next long run
must use a regenerated dataset in a new directory.

The detailed interpretation and prioritized performance plan are in
[docs/mechanics-v2-results-and-next-steps.md](docs/mechanics-v2-results-and-next-steps.md).

## Interaction-policy design schema

The raw replay archive was checked directly. A replay contains ordered `states`
and `actions` arrays, and each state records the immediately preceding player
and opponent move. Prepared training remains one row per decision, but a new
preparer can derive history events and stable revealed rosters by walking only
states at or before that decision. The legacy prepared JSONL discarded the
accumulated context; the source archive did not.

The proposed schema is documented in
[docs/interaction-policy-v3.md](docs/interaction-policy-v3.md). It fixes the
prepared-row version, cache arrays, 30-token structured layout, four-layer
interaction encoder, legal hierarchy, optional value loss, Qwen ablations, and
acceptance tests before implementation begins.

## Direct win-optimization pipeline

The project now has a Qwen-only learning path for the metric that matters:
complete battle wins. The existing interaction checkpoint is retained as the
behavior-cloning warm start. A new action-value head learns the logged action's
terminal outcome, the state value uses expectile regression, and the policy is
updated with advantage-weighted imitation before self-play begins.

Local Showdown self-play records sampled legal actions, exact old log
probabilities, centered values, terminal rewards, GAE advantages, and returns.
The PPO updater uses policy and value clipping, entropy regularization, gradient
clipping, and a target-KL stop. A persistent league samples frozen Qwen
checkpoints and promotes a candidate only through complete-game evaluation.
Rejected models remain on disk.

The first pilot command is:

```bash
python -m pokemon_battler.win_experiment \
  --output-dir outputs/qwen-win-pilot-1
```

No search engine or external opponent chooses actions in this pipeline.
Showdown is used only for battle mechanics and observations. The default team
fixture is expanded into six lead rotations; multiple genuine team files can
be supplied for broader training. The implementation and limitations are in
[docs/qwen-win-training.md](docs/qwen-win-training.md).

## Public Showdown runner and online outcome batches

Public account play is now wired through `pokemon_battler.public_play`. It
loads the registered username and password from an ignored `.env`, provides a
cheap login-only probe, accepts or sends bounded allowlisted challenges, and
retains the public state/action/result stream as PPO-compatible JSONL. The
default public checkpoint follows
`outputs/qwen-win-pilot-1/selected_checkpoint.txt`.

Optional `--learn` mode samples one frozen champion for a complete public
batch, rejects incomplete or fallback-contaminated trajectories, trains a new
checkpoint between games, and runs local candidate-versus-champion promotion.
Rejected candidates stay on disk. Public team preview is randomized rather
than permanently leading team slot one, but preview itself is not yet a learned
action. Usage and commands are documented in
[docs/public-showdown-learning.md](docs/public-showdown-learning.md).
