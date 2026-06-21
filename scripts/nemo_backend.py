from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NemoProcessingResult:
    text: str
    keep: bool
    steps: list[str]
    scores: dict[str, float | int | bool]
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

    def __init__(self) -> None:
        try:
            import pandas as pd
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
                UnicodeReformatter(normalization="NFC", explain=False),
                NewlineNormalizer(),
            ],
            input_fields="text",
        )
        self._filters = [
            WordCountFilter(min_words=10, max_words=100000, lang="en"),
            UrlsFilter(max_url_to_text_ratio=0.30),
            NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.80),
            WhiteSpaceFilter(max_white_space_ratio=0.50),
        ]
        self.modifier_steps = [
            "DocumentBatch[pandas_dataframe]",
            "Modify[UnicodeReformatter(normalization=NFC)]",
            "Modify[NewlineNormalizer]",
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
