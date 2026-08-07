from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from pokemon_battler.actions import ACTION_COUNT, action_label, legal_action_ids
from pokemon_battler.prompting import encode_candidate_prompt, render_prompt

POLICY_HEAD_FILENAME = "policy_head.safetensors"
CANDIDATE_HEAD_FILENAME = "candidate_head.safetensors"


class CandidateHead(torch.nn.Module):
    """Apply one shared scoring function to every legal candidate representation."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        intermediate_size = max(hidden_size // 2, 64)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, intermediate_size),
            torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states).squeeze(-1)


def _resolve_cached_path(model_name_or_path: str, local_files_only: bool) -> str:
    path = Path(model_name_or_path)
    if path.exists() or not local_files_only:
        return model_name_or_path
    try:
        from huggingface_hub import try_to_load_from_cache

        config_path = try_to_load_from_cache(model_name_or_path, "config.json")
        if isinstance(config_path, str) and Path(config_path).is_file():
            return str(Path(config_path).parent)
        raise FileNotFoundError(model_name_or_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"No complete local Hugging Face snapshot was found for {model_name_or_path!r}"
        ) from exc


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def load_tokenizer(model_name_or_path: str, *, local_files_only: bool = False) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(
    model_name_or_path: str,
    *,
    adapter_path: str | None = None,
    dtype: str = "auto",
    load_in_4bit: bool = False,
    for_training: bool = False,
    local_files_only: bool = False,
    attn_implementation: str = "auto",
) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, __version__

    if adapter_path:
        try:
            from peft import PeftConfig
        except ImportError as exc:
            raise RuntimeError("Loading a LoRA adapter requires the 'peft' package") from exc
        adapter_config = PeftConfig.from_pretrained(
            adapter_path,
            local_files_only=local_files_only,
        )
        base_model_name = adapter_config.base_model_name_or_path
        tokenizer_source = adapter_path
    else:
        base_model_name = model_name_or_path
        tokenizer_source = model_name_or_path

    base_model_name = _resolve_cached_path(base_model_name, local_files_only)
    tokenizer_source = _resolve_cached_path(tokenizer_source, local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_dtype(dtype, device)
    dtype_argument = "dtype" if int(__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
    load_kwargs: dict[str, Any] = {
        dtype_argument: torch_dtype,
        "local_files_only": local_files_only,
    }
    if attn_implementation != "auto":
        load_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit loading requires a CUDA device")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": device.index or 0}

    tokenizer = load_tokenizer(tokenizer_source, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=for_training,
            local_files_only=local_files_only,
        )

    if not load_in_4bit:
        model.to(device)
    return model, tokenizer, device


def attach_lora(
    model: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    is_4bit: bool,
) -> Any:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("LoRA training requires the 'peft' package") from exc

    if is_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    return get_peft_model(model, config)


def indexed_logits_parameter(model: Any) -> str | None:
    """Return the model argument that accepts explicit logit sequence positions."""
    signature_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    forward_parameters = inspect.signature(signature_model.forward).parameters
    if "logits_to_keep" in forward_parameters:
        return "logits_to_keep"
    return None


def assistant_only_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    logits_parameter: str | None,
) -> torch.Tensor:
    """
    Compute causal-LM loss without projecting ignored prompt positions to vocabulary.

    The collator masks every prompt label with -100. Qwen can accept an explicit
    tensor of sequence positions for its LM head, so only positions predicting the
    action and EOS need full-vocabulary logits. This is mathematically equivalent
    to the standard shifted causal-LM loss while using much less memory.
    """
    if logits_parameter is None:
        return model(**batch).loss

    labels = batch["labels"]
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    target_positions = torch.nonzero(
        (shift_labels != -100).any(dim=0),
        as_tuple=False,
    ).flatten()
    if target_positions.numel() == 0:
        raise ValueError("Batch contains no supervised action tokens")

    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    output = model(
        **model_inputs,
        **{logits_parameter: target_positions},
    )
    selected_labels = shift_labels.index_select(1, target_positions)
    return torch.nn.functional.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        selected_labels.reshape(-1).to(output.logits.device),
        ignore_index=-100,
    )


def create_policy_head(model: Any, device: torch.device) -> torch.nn.Linear:
    hidden_size = int(model.config.hidden_size)
    head = torch.nn.Linear(hidden_size, ACTION_COUNT, bias=True, device=device)
    torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(head.bias)
    return head


def create_candidate_head(model: Any, device: torch.device) -> CandidateHead:
    head = CandidateHead(int(model.config.hidden_size)).to(device=device, dtype=torch.float32)
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return head


def save_policy_head(head: torch.nn.Linear, output_dir: str | Path) -> None:
    output_path = Path(output_dir) / POLICY_HEAD_FILENAME
    state = {
        key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key, value in head.state_dict().items()
    }
    save_safetensors(state, output_path)


def load_policy_head(
    model: Any,
    adapter_path: str | Path,
    device: torch.device,
) -> torch.nn.Linear:
    path = Path(adapter_path) / POLICY_HEAD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Policy-head checkpoint does not exist: {path}")
    head = create_policy_head(model, device)
    head.load_state_dict(load_safetensors(path, device=str(device)))
    return head


def has_policy_head(adapter_path: str | Path | None) -> bool:
    return bool(adapter_path) and (Path(adapter_path) / POLICY_HEAD_FILENAME).is_file()


def save_candidate_head(head: CandidateHead, output_dir: str | Path) -> None:
    output_path = Path(output_dir) / CANDIDATE_HEAD_FILENAME
    state = {
        key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key, value in head.state_dict().items()
    }
    save_safetensors(state, output_path)


def load_candidate_head(
    model: Any,
    adapter_path: str | Path,
    device: torch.device,
) -> CandidateHead:
    path = Path(adapter_path) / CANDIDATE_HEAD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Candidate-head checkpoint does not exist: {path}")
    head = create_candidate_head(model, device)
    head.load_state_dict(load_safetensors(path, device=str(device)))
    return head


def has_candidate_head(adapter_path: str | Path | None) -> bool:
    return bool(adapter_path) and (Path(adapter_path) / CANDIDATE_HEAD_FILENAME).is_file()


def _last_hidden_states(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    logits_parameter: str | None,
) -> torch.Tensor:
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask"}
    }
    causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    transformer = getattr(causal_model, "model", None)
    if transformer is not None:
        output = transformer(**model_inputs, use_cache=False)
        return output.last_hidden_state
    forward_kwargs: dict[str, Any] = {
        "output_hidden_states": True,
        "use_cache": False,
    }
    if logits_parameter is not None:
        forward_kwargs[logits_parameter] = 1
    output = model(**model_inputs, **forward_kwargs)
    return output.hidden_states[-1]


def masked_policy_logits(
    model: Any,
    policy_head: torch.nn.Linear,
    batch: dict[str, torch.Tensor],
    *,
    logits_parameter: str | None,
) -> torch.Tensor:
    """Return 13 action logits with illegal entries removed from the distribution."""
    # LoRA layers live inside the transformer, so this bypasses the large LM head
    # without bypassing the adapters.
    hidden_states = _last_hidden_states(
        model,
        batch,
        logits_parameter=logits_parameter,
    )
    last_positions = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    pooled = hidden_states[rows, last_positions]
    logits = policy_head(pooled.float())
    legal_mask = batch["legal_action_mask"].to(device=logits.device, dtype=torch.bool)
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("Every policy-head example must contain a legal action")
    return logits.masked_fill(~legal_mask, float("-inf"))


def masked_candidate_logits(
    model: Any,
    candidate_head: CandidateHead,
    batch: dict[str, torch.Tensor],
    *,
    logits_parameter: str | None,
) -> torch.Tensor:
    """Score action-line representations with one shared candidate network."""
    hidden_states = _last_hidden_states(
        model,
        batch,
        logits_parameter=logits_parameter,
    )
    positions = batch["candidate_positions"].to(hidden_states.device)
    legal_mask = batch["legal_action_mask"].to(hidden_states.device, dtype=torch.bool)
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("Every candidate-head example must contain a legal action")
    if bool((positions[legal_mask] < 0).any()):
        raise ValueError("A legal candidate is missing its hidden-state position")
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)[:, None]
    candidate_hidden = hidden_states[rows, positions.clamp_min(0)]
    logits = candidate_head(candidate_hidden.float())
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.shape != legal_mask.shape:
        raise ValueError("Candidate head must return one score per action candidate")
    return logits.masked_fill(~legal_mask, float("-inf"))


def policy_head_loss(
    model: Any,
    policy_head: torch.nn.Linear,
    batch: dict[str, torch.Tensor],
    *,
    logits_parameter: str | None,
) -> torch.Tensor:
    logits = masked_policy_logits(
        model,
        policy_head,
        batch,
        logits_parameter=logits_parameter,
    )
    targets = batch["action_ids"].to(logits.device)
    if not bool(torch.isfinite(logits.gather(1, targets[:, None])).all()):
        raise ValueError("A policy-head target is absent from its legal-action mask")
    return torch.nn.functional.cross_entropy(logits, targets)


def candidate_head_loss(
    model: Any,
    candidate_head: CandidateHead,
    batch: dict[str, torch.Tensor],
    *,
    logits_parameter: str | None,
) -> torch.Tensor:
    logits = masked_candidate_logits(
        model,
        candidate_head,
        batch,
        logits_parameter=logits_parameter,
    )
    targets = batch["action_ids"].to(logits.device)
    if not bool(torch.isfinite(logits.gather(1, targets[:, None])).all()):
        raise ValueError("A candidate-head target is absent from its legal-action mask")
    return torch.nn.functional.cross_entropy(logits, targets)


@torch.inference_mode()
def score_policy_head_actions(
    model: Any,
    tokenizer: Any,
    policy_head: torch.nn.Linear,
    state: dict[str, Any],
    device: torch.device,
    *,
    max_length: int = 4096,
    prompt_format: str = "verbose-v1",
) -> dict[int, float]:
    prompt_ids = tokenizer.encode(
        render_prompt(state, prompt_format),
        add_special_tokens=True,
    )
    if len(prompt_ids) > max_length:
        raise ValueError(
            f"Prompt has {len(prompt_ids)} tokens and exceeds max_length={max_length}"
        )
    legal = legal_action_ids(state)
    legal_mask = torch.zeros((1, ACTION_COUNT), dtype=torch.bool, device=device)
    legal_mask[0, legal] = True
    batch = {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
        "legal_action_mask": legal_mask,
    }
    logits = masked_policy_logits(
        model,
        policy_head,
        batch,
        logits_parameter=indexed_logits_parameter(model),
    )[0]
    return {action_id: float(logits[action_id].item()) for action_id in legal}


@torch.inference_mode()
def score_candidate_head_actions(
    model: Any,
    tokenizer: Any,
    candidate_head: CandidateHead,
    state: dict[str, Any],
    device: torch.device,
    *,
    max_length: int = 4096,
    prompt_format: str = "compact-v1",
) -> dict[int, float]:
    input_ids, positions_by_action = encode_candidate_prompt(
        tokenizer,
        state,
        prompt_format,
    )
    if len(input_ids) > max_length:
        raise ValueError(
            f"Prompt has {len(input_ids)} tokens and exceeds max_length={max_length}"
        )
    legal = legal_action_ids(state)
    legal_mask = torch.zeros((1, ACTION_COUNT), dtype=torch.bool, device=device)
    legal_mask[0, legal] = True
    positions = torch.full((1, ACTION_COUNT), -1, dtype=torch.long, device=device)
    for action_id, position in positions_by_action.items():
        positions[0, action_id] = position
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "legal_action_mask": legal_mask,
        "candidate_positions": positions,
    }
    logits = masked_candidate_logits(
        model,
        candidate_head,
        batch,
        logits_parameter=indexed_logits_parameter(model),
    )[0]
    return {action_id: float(logits[action_id].item()) for action_id in legal}


@torch.inference_mode()
def score_legal_actions(
    model: Any,
    tokenizer: Any,
    state: dict[str, Any],
    device: torch.device,
    *,
    max_length: int = 4096,
    prompt_format: str = "verbose-v1",
) -> dict[int, float]:
    """
    Score only complete legal action strings.

    Only replay-recoverable actions are candidates. A live simulator must provide
    the stricter mask for trapping, disabled moves, choice locks, and similar rules.
    Sequence log-probability includes EOS so labels such as A1 and A10 remain distinct.
    """
    prompt_ids = tokenizer.encode(
        render_prompt(state, prompt_format),
        add_special_tokens=True,
    )
    legal = legal_action_ids(state)
    candidates: list[tuple[int, list[int]]] = []
    for action_id in legal:
        candidate_ids = tokenizer.encode(action_label(action_id), add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            candidate_ids = [*candidate_ids, tokenizer.eos_token_id]
        if len(prompt_ids) + len(candidate_ids) > max_length:
            raise ValueError(
                f"Prompt and candidate exceed max_length={max_length}; "
                "increase the evaluation limit."
            )
        candidates.append((action_id, candidate_ids))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    max_candidate_length = max(len(ids) for _, ids in candidates)
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for _, candidate_ids in candidates:
        sequence = [*prompt_ids, *candidate_ids]
        padding = max_candidate_length - len(candidate_ids)
        sequences.append(sequence + [pad_token_id] * padding)
        masks.append([1] * len(sequence) + [0] * padding)

    input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    scores: dict[int, float] = {}
    limited_logits_name = indexed_logits_parameter(model)
    if limited_logits_name is None:
        signature_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        forward_parameters = inspect.signature(signature_model.forward).parameters
        if "num_logits_to_keep" in forward_parameters:
            limited_logits_name = "num_logits_to_keep"

    if limited_logits_name is not None:
        # We need the logit immediately before the first target plus one logit
        # per possible candidate token. Restricting the LM head to this suffix
        # avoids materializing [legal_actions, prompt_length, vocabulary] logits.
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{limited_logits_name: max_candidate_length + 1},
        )
        target_rows = [
            candidate_ids + [pad_token_id] * (max_candidate_length - len(candidate_ids))
            for _, candidate_ids in candidates
        ]
        target_tokens = torch.tensor(target_rows, dtype=torch.long, device=device)
        token_log_probs = torch.log_softmax(
            output.logits[:, :max_candidate_length, :].float(),
            dim=-1,
        )
        selected = token_log_probs.gather(
            -1,
            target_tokens.unsqueeze(-1),
        ).squeeze(-1)
        for row_index, (action_id, candidate_ids) in enumerate(candidates):
            scores[action_id] = float(selected[row_index, : len(candidate_ids)].sum().item())
    else:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        token_log_probs = torch.log_softmax(output.logits[:, :-1, :].float(), dim=-1)
        next_tokens = input_ids[:, 1:]
        selected = token_log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        first_target_prediction = len(prompt_ids) - 1
        for row_index, (action_id, candidate_ids) in enumerate(candidates):
            last_target_prediction = first_target_prediction + len(candidate_ids)
            score = selected[
                row_index,
                first_target_prediction:last_target_prediction,
            ].sum()
            scores[action_id] = float(score.item())
    return scores


def load_training_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path) / "training_config.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
