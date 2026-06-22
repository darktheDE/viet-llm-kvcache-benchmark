from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ftfy import fix_text

try:
    from scripts.utils_text import (
        has_replacement_char,
        normalize_nfc,
        normalize_whitespace,
        remove_control_chars,
        text_quality_flags,
    )
except ModuleNotFoundError:
    from utils_text import (
        has_replacement_char,
        normalize_nfc,
        normalize_whitespace,
        remove_control_chars,
        text_quality_flags,
    )


@dataclass
class NemoProcessingResult:
    text: str
    keep: bool
    steps: list[str]
    scores: dict[str, Any]
    rejection_reason: str | None = None


class NemoBackendError(RuntimeError):
    """Raised when the installed NeMo Curator API cannot be used."""


class NemoTextBackend:
    """Small adapter around NeMo Curator 1.2.0 text stages.

    The installed `nemo-curator[text-cpu]==1.2.0` exposes text modifiers and
    filters as ProcessingStage/DocumentFilter classes. This adapter keeps the
    project pipeline independent from local containers while still exercising
    the real NeMo Curator API.
    """

    def __init__(self, min_chars: int = 200) -> None:
        try:
            import pandas as pd
            from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
            from nemo_curator.stages.text.filters.heuristic.string import (
                NonAlphaNumericFilter,
                UrlsFilter,
                WhiteSpaceFilter,
                WordCountFilter,
            )
            from nemo_curator.stages.text.modifiers.modifier import Modify
            from nemo_curator.stages.text.modifiers.string.newline_normalizer import (
                NewlineNormalizer,
            )
            from nemo_curator.stages.text.modifiers.unicode.unicode_reformatter import (
                UnicodeReformatter,
            )
            from nemo_curator.tasks.document import DocumentBatch
        except Exception as exc:  # pragma: no cover - depends on container package state.
            raise NemoBackendError(f"Cannot import NeMo Curator text API: {exc}") from exc

        self._pd = pd
        self._document_batch_cls = DocumentBatch
        self._modifier_stage = Modify(
            [
                _project_ftfy_fix_text,
                UnicodeReformatter(normalization="NFC", explain=False),
                NewlineNormalizer(),
                _project_text_postprocess,
            ],
            input_fields="text",
        )
        self._filters = [
            WordCountFilter(min_words=10, max_words=100000, lang="en"),
            UrlsFilter(max_url_to_text_ratio=0.30),
            NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.80),
            WhiteSpaceFilter(max_white_space_ratio=0.50),
        ] + _build_project_filters(DocumentFilter, min_chars=min_chars)
        self.modifier_steps = [
            "DocumentBatch[pandas_dataframe]",
            "Modify[ProjectFtfyFixText]",
            "Modify[UnicodeReformatter(normalization=NFC)]",
            "Modify[NewlineNormalizer]",
            "Modify[ProjectTextPostprocessor(normalization=NFC,control_char_removal,whitespace_normalization)]",
        ]
        self.filter_steps = [
            f"DocumentFilter[{filter_obj.name}:{filter_obj.__class__.__name__}]"
            for filter_obj in self._filters
        ]
        self.steps = self.modifier_steps + self.filter_steps

    def process_texts(self, texts: list[str], dataset_name: str = "raw_records") -> list[NemoProcessingResult]:
        if not texts:
            return []

        data = self._pd.DataFrame({"text": [text or "" for text in texts]})
        batch = self._document_batch_cls(task_id="clean_with_nemo", dataset_name=dataset_name, data=data)
        modified_batch = self._modifier_stage.process(batch)
        if modified_batch is None:
            raise NemoBackendError("NeMo Modify.process returned None")

        modified_data = modified_batch.to_pandas()
        results: list[NemoProcessingResult] = []
        for text in modified_data["text"].tolist():
            keep = True
            rejection_reason = None
            scores: dict[str, float | int | bool] = {}

            for filter_obj in self._filters:
                score = filter_obj.score_document(text)
                keep_filter = bool(filter_obj.keep_document(score))
                score_key = f"{filter_obj.name}_score"
                keep_key = f"{filter_obj.name}_keep"
                scores[score_key] = _jsonable_scalar(score)
                scores[keep_key] = keep_filter
                if not keep_filter:
                    keep = False
                    rejection_reason = filter_obj.name
                    break

            results.append(
                NemoProcessingResult(
                    text=text,
                    keep=keep,
                    steps=self.steps,
                    scores=scores,
                    rejection_reason=rejection_reason,
                )
            )
        return results


def _jsonable_scalar(value: Any) -> float | int | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except Exception:
        return 0.0


def _project_ftfy_fix_text(text: str) -> str:
    return fix_text(text or "")


def _project_text_postprocess(text: str) -> str:
    normalized = normalize_nfc(text or "")
    no_control = remove_control_chars(normalized)
    return normalize_whitespace(no_control)


def _build_project_filters(document_filter_cls: type[Any], min_chars: int) -> list[Any]:
    class MinCharacterCountFilter(document_filter_cls):
        def __init__(self, min_chars: int) -> None:
            super().__init__()
            self._min_chars = min_chars
            self._name = "min_characters"

        def score_document(self, text: str) -> int:
            return len(text or "")

        def keep_document(self, score: int) -> bool:
            return score >= self._min_chars

    class ReplacementCharacterFilter(document_filter_cls):
        def __init__(self) -> None:
            super().__init__()
            self._name = "replacement_char_free"

        def score_document(self, text: str) -> int:
            return int(not has_replacement_char(text or ""))

        def keep_document(self, score: int) -> bool:
            return bool(score)

    class LetterRatioFilter(document_filter_cls):
        def __init__(self, min_letter_ratio: float = 0.45) -> None:
            super().__init__()
            self._min_letter_ratio = min_letter_ratio
            self._name = "letter_ratio"

        def score_document(self, text: str) -> float:
            return float(text_quality_flags(text or "")["letter_ratio"])

        def keep_document(self, score: float) -> bool:
            return score >= self._min_letter_ratio

    class StrangeSymbolRatioFilter(document_filter_cls):
        def __init__(self, max_symbol_ratio: float = 0.20) -> None:
            super().__init__()
            self._max_symbol_ratio = max_symbol_ratio
            self._name = "strange_symbol_ratio"

        def score_document(self, text: str) -> float:
            return float(text_quality_flags(text or "")["symbol_ratio"])

        def keep_document(self, score: float) -> bool:
            return score <= self._max_symbol_ratio

    class VietnameseSignalFilter(document_filter_cls):
        def __init__(self) -> None:
            super().__init__()
            self._name = "vietnamese_signal"

        def score_document(self, text: str) -> int:
            return int(bool(text_quality_flags(text or "")["has_vietnamese_signal"]))

        def keep_document(self, score: int) -> bool:
            return bool(score)

    return [
        MinCharacterCountFilter(min_chars=min_chars),
        ReplacementCharacterFilter(),
        LetterRatioFilter(),
        StrangeSymbolRatioFilter(),
        VietnameseSignalFilter(),
    ]
