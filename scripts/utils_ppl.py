from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - used only in lightweight local envs
    class _TorchShim:
        class cuda:  # type: ignore[no-redef]
            class OutOfMemoryError(Exception):
                pass

            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

        class inference_mode:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

    torch = _TorchShim()  # type: ignore[assignment]


def _infer_max_length(reference_model: Any, tokenizer: Any, max_length: int | None) -> int:
    if max_length is not None and max_length > 0:
        return max_length

    config = getattr(reference_model, "config", None)
    for attr in ("n_positions", "max_position_embeddings", "seq_length", "model_max_length"):
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and 0 < value < 10**9:
            return value

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 10**9:
        return tokenizer_max

    return 2048


def _empty_result(status: str, error: str = "") -> dict[str, Any]:
    return {
        "perplexity": None,
        "ppl_loss": None,
        "ppl_tokens": 0,
        "ppl_status": status,
        "ppl_error": error,
    }


def _tokenize(tokenizer: Any, text: str, device: Any) -> Any:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"].to(device)


def compute_perplexity(
    reference_model: Any,
    tokenizer: Any,
    generated_text: str,
    device: Any,
    prompt_text: str = "",
    mode: str = "conditional",
    max_length: int | None = None,
    stride: int = 512,
) -> dict[str, Any]:
    """Compute PPL with a reference model, never with the compressed benchmark model.

    mode="conditional" scores generated tokens conditioned on prompt_text.
    mode="generated_only" scores only generated_text as a standalone sequence.
    """

    if not generated_text or not str(generated_text).strip():
        return _empty_result("EMPTY", "empty generated_text")

    if reference_model is None or tokenizer is None:
        return _empty_result("ERROR", "reference model or tokenizer not available")

    if mode not in {"conditional", "generated_only"}:
        return _empty_result("ERROR", f"unsupported ppl mode: {mode}")

    try:
        reference_model.eval()
        resolved_max_length = _infer_max_length(reference_model, tokenizer, max_length)
        stride = max(1, min(int(stride), resolved_max_length))

        generated_ids = _tokenize(tokenizer, generated_text, device)
        generated_len = int(generated_ids.size(1))
        if generated_len == 0:
            return _empty_result("EMPTY", "no generated tokens after tokenization")

        if mode == "conditional" and prompt_text and str(prompt_text).strip():
            prompt_ids = _tokenize(tokenizer, prompt_text, device)
            prompt_len = int(prompt_ids.size(1))
            input_ids = torch.cat([prompt_ids, generated_ids], dim=1)
            score_start = prompt_len
        else:
            input_ids = generated_ids
            score_start = 0

        seq_len = int(input_ids.size(1))
        if seq_len <= 1:
            return _empty_result("EMPTY", "sequence too short for perplexity")

        score_end = seq_len
        total_nll = 0.0
        total_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + resolved_max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            if trg_len <= 0:
                continue

            input_window = input_ids[:, begin_loc:end_loc]
            if int(input_window.size(1)) <= 1:
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            labels = input_window.clone()
            positions = torch.arange(begin_loc, end_loc, device=input_window.device)
            score_mask = (positions >= score_start) & (positions < score_end)
            labels[:, ~score_mask] = -100
            labels[:, :-trg_len] = -100

            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens <= 0:
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            with torch.inference_mode():
                outputs = reference_model(
                    input_ids=input_window,
                    attention_mask=torch.ones_like(input_window),
                    labels=labels,
                )

            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError("reference model did not return loss")

            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                raise OverflowError("loss is not finite")
            if loss_value > 50:
                return {
                    "perplexity": None,
                    "ppl_loss": loss_value,
                    "ppl_tokens": total_tokens,
                    "ppl_status": "ERROR",
                    "ppl_error": "loss overflow",
                }

            total_nll += loss_value * valid_tokens
            total_tokens += valid_tokens
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if total_tokens <= 0:
            return _empty_result("EMPTY", "no scoreable generated tokens")

        mean_nll = total_nll / total_tokens
        if not math.isfinite(mean_nll) or mean_nll > 50:
            return {
                "perplexity": None,
                "ppl_loss": mean_nll if math.isfinite(mean_nll) else None,
                "ppl_tokens": total_tokens,
                "ppl_status": "ERROR",
                "ppl_error": "mean loss overflow",
            }

        perplexity = math.exp(mean_nll)
        if not math.isfinite(perplexity):
            return {
                "perplexity": None,
                "ppl_loss": mean_nll,
                "ppl_tokens": total_tokens,
                "ppl_status": "ERROR",
                "ppl_error": "perplexity overflow",
            }

        return {
            "perplexity": round(perplexity, 4),
            "ppl_loss": round(mean_nll, 6),
            "ppl_tokens": total_tokens,
            "ppl_status": "OK",
            "ppl_error": "",
        }

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "OOM",
            "ppl_error": str(exc),
        }
    except Exception as exc:
        return {
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "ERROR",
            "ppl_error": str(exc),
        }


def load_reference_model(
    model_name: str,
    tokenizer_name: str | None = None,
    device: str = "cuda",
    dtype: str = "bf16",
) -> tuple[Any, Any, Any]:
    """Load the BF16/FP reference model used for offline PPL scoring."""

    import torch as real_torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bf16": real_torch.bfloat16,
        "fp16": real_torch.float16,
        "fp32": real_torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, real_torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if device == "auto":
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        reference_model.to(device)

    reference_model.eval()
    resolved_device = next(reference_model.parameters()).device
    return reference_model, tokenizer, resolved_device
