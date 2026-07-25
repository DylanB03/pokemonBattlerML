# Pokémon Battler Roadmap

## Vision

Build an uncertainty-aware competitive Pokémon agent and post-game coach that
can:

1. Play complete Pokémon Showdown battles as a disclosed bot or practice
   opponent.
2. Show the policy model's preference across every legal action.
3. Review a replay using only the information available to the player at each
   decision.
4. Estimate the counterfactual match-win probability of alternative actions.
5. Distinguish questionable decisions from bad luck and genuinely close calls.
6. Use a stronger language model to explain calculated evidence and generate
   personalized training exercises.

The review system is the long-term differentiator. The language model should
explain evidence produced by the policy, value model, and simulator; it should
not invent move grades or win probabilities by itself.

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
- Full SFT, LoRA, and 4-bit QLoRA support.
- A policy scorer that returns a score for every legal action.
- Offline evaluation against recorded human actions.

Not yet implemented:

- Preference percentages or probability calibration.
- A complete Pokémon Showdown battle client.
- Per-turn decision traces covering an entire battle.
- A replay-review interface.
- A state-value or action-value model.
- Counterfactual simulator analysis and regret.
- Grounded natural-language coaching.

## Phase 0 — Validate the behavior-cloned policy

Status: in progress.

Deliverables:

- Complete the high-rated Generation 9 OU development dataset.
- Pass the 128-example memorization test.
- Run the planned hyperparameter search.
- Train and select the first main SFT checkpoint.
- Report held-out action agreement, action-type accuracy, legality, and
  top-action margins.
- Evaluate the selected policy in simulated battles against fixed opponents.

Exit criteria:

- The trained checkpoint clearly improves over the base model on held-out
  action agreement.
- It always selects a legal action.
- Battle evaluation is reproducible and reported with confidence intervals.

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
- Results are robust enough to changes in sampled hidden information and
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
