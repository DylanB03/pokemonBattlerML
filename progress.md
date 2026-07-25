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

Verified locally with:

- 11 unit tests.
- One real Qwen2.5-0.5B LoRA optimization and adapter-save step.
- Adapter reload and constrained action evaluation.

Next experiment: prepare a high-rated Gen 9 OU pilot, inspect prompt lengths, run the
128-example overfit check, and then train the first 50k-example model.
