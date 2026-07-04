from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
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
        if isinstance(value, int) and value > 0:
            return value

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and tokenizer_max > 0 and tokenizer_max < 10**9:
        return tokenizer_max

    return 2048


def compute_perplexity(
    reference_model: Any,
    tokenizer: Any,
    text: str,
    device: Any,
    max_length: int | None = None,
    stride: int = 512,
) -> dict[str, Any]:
    """Compute perplexity from a BF16 reference model."""

    if not text or not str(text).strip():
        return {
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "EMPTY",
            "ppl_error": "empty text",
        }

    if reference_model is None or tokenizer is None:
        return {
            "perplexity": None,
            "ppl_loss": None,
            "ppl_tokens": 0,
            "ppl_status": "ERROR",
            "ppl_error": "reference model or tokenizer not available",
        }

    try:
        reference_model.eval()
        resolved_max_length = _infer_max_length(reference_model, tokenizer, max_length)
        stride = max(1, min(int(stride), resolved_max_length))

        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        seq_len = int(input_ids.size(1))
        if seq_len == 0:
            return {
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "EMPTY",
                "ppl_error": "no tokens after tokenization",
            }

        total_nll = 0.0
        total_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + resolved_max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            if trg_len <= 0:
                continue

            input_window = input_ids[:, begin_loc:end_loc]
            if input_window.size(1) <= 1:
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            labels = input_window.clone()
            labels[:, :-trg_len] = -100

            model_inputs = {"input_ids": input_window, "labels": labels}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask[:, begin_loc:end_loc]

            with torch.inference_mode():
                outputs = reference_model(**model_inputs)

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

            total_nll += loss_value * trg_len
            total_tokens += trg_len
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if total_tokens <= 0:
            return {
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": 0,
                "ppl_status": "EMPTY",
                "ppl_error": "sequence too short for perplexity",
            }

        mean_nll = total_nll / total_tokens
        if not math.isfinite(mean_nll):
            return {
                "perplexity": None,
                "ppl_loss": None,
                "ppl_tokens": total_tokens,
                "ppl_status": "ERROR",
                "ppl_error": "mean loss is not finite",
            }

        if mean_nll > 50:
            return {
                "perplexity": None,
                "ppl_loss": mean_nll,
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
