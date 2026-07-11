"""Shared types and lightweight helpers for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetricResult:
    name: str
    score: float
    details: Optional[List[Dict[str, Any]]] = field(default=None)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token.isalnum()}


def keyword_overlap(contexts: List[str], reference: str) -> float:
    """Share of reference tokens found in retrieved context."""
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 0.0
    merged = " ".join(contexts)
    return len(ref_tokens & tokenize(merged)) / len(ref_tokens)


def answer_length_ratio(answer: str, reference: str) -> float:
    ref_len = max(len(reference.split()), 1)
    return len(answer.split()) / ref_len


def mean_score(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0
