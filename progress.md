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
- Includes a move-name-free hybrid policy with a versioned 97-value mechanics
  tensor for each action.
- Precomputes mechanics into memory-mapped float16 caches and provides a
  mechanics-only MLP ablation.
- Stops one-command runs after four flat validation checks while preserving the
  exact best checkpoint.

Verified locally with:

- 39 unit tests.
- One real Qwen2.5-0.5B LoRA optimization and adapter-save step.
- Adapter reload and constrained action evaluation.
- One real Qwen2.5-0.5B policy-head optimization, save, reload, and prediction.
- One real Qwen2.5-0.5B candidate-head optimization plus validation, best/final
  save, reload, automatic prompt/head detection, and constrained prediction.
- Non-language-model train, save, reload, and batched evaluation smoke tests.
- Mechanics feature, cache, hybrid-score, and mechanics-only backpropagation
  tests.

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
  --output-dir outputs/mechanics-v1-1epoch
```

This builds or reuses the mechanics caches, trains for at most one complete
pass, stops a validation plateau automatically, and compares the best and final
checkpoints on the same validation rows.
