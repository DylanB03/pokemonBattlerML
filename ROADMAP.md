# Pokémon Battler Roadmap

## My goal

My goal is to build a competitive Qwen Pokémon policy whose primary objective
is match win rate. The policy should:

1. Play complete Pokémon Showdown battles as a disclosed bot or practice
   opponent.
2. Show the policy model's preference across every legal action.
3. Review a replay using only the information available to the player at each
   decision.
4. Estimate the counterfactual match-win probability of alternative actions.
5. Distinguish questionable decisions from bad luck and genuinely close calls.
6. Improve through outcome-aware replay training and self-play reinforcement
   learning without delegating decisions to a search engine.

Replay review remains a possible application, but it is secondary to producing
a strong direct policy. Showdown supplies rules, observations, and results. It
does not supply the policy with recommended moves.

## Metric definitions

These quantities must remain separate in code, reports, and user-facing text.

### Policy score

The current causal language model assigns each complete legal action string a
sequence log-probability. This is the raw score already returned by
`score_legal_actions`.

### Model preference

Normalize the legal-action scores into a distribution:

```text
preference(a) = exp(score(a)) / sum(exp(score(b)) for legal action b)
```

The resulting percentages answer:

> How strongly does the behavior-cloned policy prefer each legal action?

They do not represent match-win probability. Even after calibration, they
measure agreement with actions represented in the human training distribution,
not objective move quality.

### Match-win probability

For information available at turn `t`, `I_t`, and legal action `a`:

```text
Q(I_t, a) = P(eventually win | information at turn t, choose action a)
```

This must be estimated with a learned value model, simulator search, or a
combination of both. It must account for possible hidden opponent sets,
simultaneous opponent actions, and battle randomness.

### Estimated regret

Once counterfactual action values are sufficiently reliable:

```text
regret_t = max(Q(I_t, a)) - Q(I_t, action_played)
```

Regret is measured in estimated match-win-probability points. A move should not
receive a mistake label when alternatives are effectively tied or the estimate
is too uncertain.

## Current foundation

Implemented:

- Streaming preparation of Metamon replay trajectories.
- Battle-grouped and chronological dataset splits.
- Turn-level Generation 9 OU behavior-cloning examples.
- Legal-action validation and constrained inference.
- Full SFT, fixed policy-head, shared candidate-head, mechanics-head, LoRA, and
  4-bit QLoRA support.
- Compact, verbose, legacy move-name-free, and mechanics-v2 prompt serializers.
- Versioned 207-value candidate mechanics vectors, 32 categorical identity
  fields, memory-mapped caches, and a mechanics-only MLP ablation.
- A legally masked policy scorer that returns a score for every candidate.
- Deterministic offline ranking, family, entropy, and action-agreement evaluation.
- One-command cache preparation, plateau-aware training, best/final evaluation,
  and reporting.
- Schema-3 trajectory preparation with stable rosters and backward-only history.
- A structured interaction transformer with hierarchical action and auxiliary
  value heads, plus an end-to-end raw-archive-to-report runner.
- A local `poke-env` Showdown player that applies the live request mask, maps the
  interaction policy's A0-A12 output back to battle orders, and records complete
  decision traces and win-rate summaries.
- Pinned, process-isolated integrations for PokéChamp One-Step, PokéChamp
  Abyssal, and Foul Play, including automatic setup and multi-game lifecycle
  handling.
- A completed fixed-team live benchmark: 20–0 against each PokéChamp heuristic
  and 4–16 against Foul Play at 100 ms, with zero fallbacks over 1,659 decisions.
- A separate per-action win-value head that leaves older interaction
  checkpoints loadable.
- Outcome-conditioned offline training of `Q(s,a)`, `V(s)`, and the legal Qwen
  policy from terminal replay results.
- Complete sampled self-play rollout capture with terminal-only rewards, GAE,
  clipped PPO, value clipping, entropy, and KL monitoring.
- A persistent frozen-Qwen league, promotion matches, non-overwriting candidate
  checkpoints, lead rotations, and a single win-training command.
- Schema-4 whole-POV preparation with consecutive next states, semi-Markov gaps,
  small transition rewards, and terminal-only dominant win/loss rewards.
- A memory-mapped frozen Qwen-plus-interaction representation cache and matched
  memoryless/recurrent candidate-policy comparison.
- Two-critic next-state IQL with recurrent burn-in, per-battle live GRU state,
  duplicate-request protection, and paired complete-game selection.
- A public Showdown runner with `.env` account loading, challenge and ladder
  modes, concurrent frozen battles, resumable reports, and optional
  between-batch PPO.
- A completed 1,000-game frozen public measurement for the Metamon v2
  checkpoint: 502 wins, 498 losses, and zero fallbacks across 30,385 decisions.

Not yet implemented:

- Probability calibration and a user-facing preference display. Live traces
  already record normalized legal-action preferences.
- A replay-review interface.
- Calibration evidence for the new state/action values across a broad team and
  opponent distribution. The estimators now exist but are not calibrated by
  implementation alone.
- Counterfactual simulator analysis and regret.
- Grounded natural-language coaching.
- A learned team-preview policy. The first self-play pipeline rotates all six
  submitted leads and accepts multiple team files, but Qwen's PPO action space
  currently begins at the first battle turn.
- GPU request batching across concurrent Showdown battles.

## Phase 0.5 — Optimize complete-game wins

Status: implemented and evaluated. The later large-data sidecar is now the
strongest measured policy.

The `pokemon_battler.win_experiment` runner uses the selected behavior-cloning
checkpoint as a warm start, learns outcome-conditioned action and state values,
collects Qwen-versus-frozen-Qwen games, applies PPO, and gates every candidate
against the current champion. The only environmental reward is the final
win/loss result. Search policies and mechanics engines are not in the training
or inference path.

The first run tested whether promotion win rate improved while action legality
remained perfect. Later scaling work moved to broader opponent and team
populations before considering a Qwen size change. The exact command,
artifacts, and limits are documented in
[Training Qwen for battle wins](docs/qwen-win-training.md).

## Phase 0 — Validate the behavior-cloned policy

Status: complete for the initial fixed-team gate. Mechanics-v2 and
interaction-policy training are complete. The final interaction checkpoint
reached 43.92% exact agreement on the fixed 5,000-row validation sample. The
local harness completed 60 measured battles against three pinned external
opponents with no live-policy fallback.

Deliverables:

- Complete the high-rated Generation 9 OU development dataset.
- Pass the 128-example memorization test.
- Preserve the step-5,000 candidate-head checkpoint as a 36.52% validation
  reference, not as a completed epoch or live-play result.
- Preserve the completed mechanics-v2 checkpoint: 42.86% exact action agreement,
  64.78% top-2, and 78.78% top-3 on 5,000 validation rows.
- Regenerate the development split with preparation schema 3. The dataset used
  by the completed run lacks recent history, accumulated opponent reveals, and
  legal-mask-quality metadata now supported by the preparer.
- Run the automated 128-row memorization gate for the interaction representation.
- Compare the hybrid against the mechanics-only ablation to measure the value of
  Qwen's state representation.
- Measure the implemented team/candidate interaction encoder and hierarchical
  action-family objective against mechanics-v2.
- Measure the implemented auxiliary state-value head before considering
  outcome-weighted imitation or offline reinforcement learning.
- Report held-out action agreement, action-type accuracy, replay-candidate
  constraint coverage, and top-action margins. Measure true legality only under
  the live simulator mask.
- Evaluate the selected policy in simulated battles against fixed opponents.

Exit criteria:

- The trained checkpoint clearly improves over the base model on held-out
  action agreement.
- It always selects a legal action.
- Battle evaluation is reproducible and reported with confidence intervals.

The rationale and experiment order are documented in
[What mechanics-v2 taught me, and what I changed next](docs/mechanics-v2-results-and-next-steps.md).
The implementation contract is
[Interaction policy v3](docs/interaction-policy-v3.md).

## Phase 1 — Expose model-preference percentages

Deliverables:

- Convert legal-action log scores to numerically stable percentages using
  log-sum-exp normalization.
- Return, for every legal action:
  - action ID and human-readable action;
  - raw log score;
  - model-preference percentage;
  - rank;
  - margin from the top action.
- Add optional temperature scaling fitted only on validation data.
- Report entropy or another distribution-level measure showing whether the
  policy strongly prefers one action or considers several actions plausible.

Required tests:

- Percentages sum to approximately 100% over legal actions.
- Illegal actions never appear in the distribution.
- Normalization remains stable for very negative scores.
- Adding the same constant to every raw score does not change the percentages.
- Action labels with different tokenizations remain distinct.

User-facing terminology:

- Use `model preference`, `policy preference`, or `human-policy agreement`.
- Do not call these percentages `win probability`, `move quality`, or
  `confidence that the move is correct`.

## Phase 2 — Record complete decision traces

Deliverables:

- Connect the policy to a local Pokémon Showdown battle client.
- At every decision, save:
  - the public information state;
  - all legal actions;
  - the complete preference distribution;
  - the selected action;
  - the model checkpoint and inference settings;
  - the resulting battle events.
- Store traces in a versioned JSON schema that can be replayed independently of
  the live client.
- Display preference bars while the disclosed bot or practice opponent chooses.

Exit criteria:

- A complete battle can be replayed with every decision and distribution
  aligned to the correct turn.
- Forced switches, Terastallization, forfeits, and terminal states are handled
  correctly.

For competitive integrity, live recommendations for a human player's rated
games should remain separate from bot and practice modes. Post-game review does
not require revealing recommendations before a move is locked.

## Phase 3 — Build the first replay review

This phase reviews policy agreement without claiming to identify objective
mistakes.

Deliverables:

- Import a Pokémon Showdown replay or saved decision trace.
- Reconstruct the information available before each decision without leaking
  facts revealed on later turns.
- Show:
  - the action the player chose;
  - the policy's top alternatives;
  - the preference percentage of each action;
  - the preference rank of the played action;
  - turns where the player and policy differed most;
  - turns where the policy itself was uncertain.
- Generate a concise match summary based on these descriptive metrics.

Exit criteria:

- Reviews are deterministic for a fixed checkpoint and trace.
- The interface clearly labels the output as policy comparison, not an
  objective blunder analysis.

## Phase 4 — Train and calibrate a state-value model

Deliverables:

- Train `V(I_t)`, an estimate of eventual match outcome from the information
  available at turn `t`.
- Preserve chronological and battle-grouped splits.
- Compare value-model variants and report:
  - log loss;
  - Brier score;
  - calibration curves;
  - discrimination by game phase;
  - performance under different team and matchup distributions.
- Plot a match-win-probability curve over completed battles.

Important limitation:

`V(I_t)` evaluates a state. It does not by itself reveal what would have
happened after actions that were not played.

Exit criteria:

- The value estimate is demonstrably better than simple baselines.
- Reliability plots establish where the model is calibrated and where its
  estimates should be treated as uncertain.

## Phase 5 — Counterfactual action values and regret

Deliverables:

- For each candidate action, estimate `Q(I_t, a)` using simulator continuations
  and the value model.
- Model the distribution over:
  - hidden opponent items, abilities, moves, and spreads;
  - possible opponent actions;
  - damage rolls, accuracy, critical hits, and secondary effects.
- Preserve simultaneous action selection rather than allowing the simulated
  opponent to react after seeing the player's choice.
- Produce uncertainty intervals and retain the most important counterfactual
  continuations.
- Calculate regret only from these action-value estimates.
- Establish mistake-label thresholds using empirical calibration rather than
  arbitrary percentages.

Possible review labels:

- `best or effectively tied`;
- `reasonable alternative`;
- `inaccuracy`;
- `major mistake`;
- `uncertain — insufficient evidence`.

Exit criteria:

- Counterfactual estimates outperform policy preference alone at predicting
  simulated and held-out battle outcomes.
- Near-tied actions are not systematically overgraded.
- Results remain stable under changes in sampled hidden information and
  opponent policy.

## Phase 6 — Separate decision quality from luck

Deliverables:

- Record observable chance events such as misses, critical hits, full
  paralysis, flinches, and secondary effects.
- Compare expected and realized consequences where the simulator supports it.
- Identify whether a match changed because of:
  - the player's decision;
  - the opponent's decision;
  - hidden information;
  - random outcomes;
  - a combination of these factors.
- Show decision-regret and luck summaries separately.

Exit criteria:

- The reviewer does not blame a player for an unfavorable random outcome after
  a sound decision.
- It also does not excuse a poor decision merely because the realized outcome
  happened to be favorable.

## Phase 7 — Grounded frontier-model explanations

Deliverables:

- Send the explanation model structured evidence rather than only a raw replay:
  - the information available at the decision;
  - the played action and important alternatives;
  - policy preferences;
  - estimated action values and uncertainty;
  - likely opponent responses;
  - representative simulator continuations;
  - verified mechanics and damage calculations.
- Use the expensive model only for the most consequential or user-selected
  turns.
- Require the explanation to distinguish facts, model estimates, and
  assumptions.
- Add automated checks for invented Pokémon, moves, items, and impossible
  mechanics.

Exit criteria:

- Explanations agree with their supplied numerical evidence.
- Explanations remain useful when multiple actions are close.
- Unsupported claims are rejected or explicitly marked as uncertain.

## Phase 8 — Personalized coaching

Deliverables:

- Track recurring decision patterns across a player's reviewed games.
- Group mistakes into actionable categories such as:
  - unnecessary risk;
  - failure to preserve a win condition;
  - poor Terastallization timing;
  - predictable play;
  - avoidable immunity or resistance;
  - missed setup or recovery opportunity;
  - inaccurate opponent-set assumptions.
- Rank patterns by estimated match impact rather than raw frequency.
- Generate review exercises and battle puzzles from the player's own critical
  turns.
- Track whether each targeted pattern improves over time.

Exit criteria:

- Coaching recommendations are supported by multiple reviewed decisions.
- The system gives a small number of prioritized, measurable practice goals
  rather than a generic battle summary.

## Product modes

The completed system may expose the same underlying models through separate
modes:

- **Bot mode:** the policy plays complete battles under a disclosed bot
  identity.
- **Practice mode:** a human plays against the bot and may optionally see the
  bot's preferences after actions are locked.
- **Replay review:** a human imports a completed battle for policy comparison,
  win-probability analysis, and coaching.
- **Research mode:** developers inspect full states, distributions, rollouts,
  calibration, and model comparisons.

Keeping these modes separate makes the intended use clear and prevents a
post-game training tool from quietly becoming live assistance in rated human
play.

The first public bot-mode runner is now implemented. It loads a dedicated
account from an ignored `.env`, supports login-only verification and bounded
challenge/ladder sessions, records replayable public trajectories, and can run
PPO only between frozen batches. Public candidates are retained separately and
must beat the frozen champion in local full-battle promotion games. Learned
team preview and a broad multi-team promotion suite remain future work.

## Guiding principles

- Never present policy likelihood as objective move quality.
- Never present an uncalibrated value as a precise probability.
- Evaluate decisions using information available at the time.
- Preserve uncertainty instead of forcing every turn into a best-move label.
- Treat simultaneous actions, hidden information, and randomness as core game
  mechanics.
- Use language models to explain calculated evidence, not replace it.
- Keep every review reproducible by recording model, data, simulator, and
  inference versions.
- Distinguish implemented, experimental, and planned features in all public
  documentation.
