"""Utilities for heuristic reranking."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable

from .utils import clean_text


_PUNCT_RE = re.compile(r"[.!?]")
_DIGIT_RE = re.compile(r"\d")


def get_length_stats(model: Dict[str, Any]) -> Dict[str, float]:
    stats = model.get("stats")
    if isinstance(stats, dict) and "tgt_len_mean" in stats and "tgt_len_std" in stats:
        return {
            "tgt_len_mean": float(stats.get("tgt_len_mean", 0.0)),
            "tgt_len_std": float(stats.get("tgt_len_std", 1.0) or 1.0),
        }
    train_tgt = model.get("train_tgt") or []
    lengths = [len(clean_text(str(text))) for text in train_tgt]
    if not lengths:
        return {"tgt_len_mean": 0.0, "tgt_len_std": 1.0}
    mean = sum(lengths) / len(lengths)
    var = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    std = math.sqrt(var) if var > 0 else 1.0
    return {"tgt_len_mean": float(mean), "tgt_len_std": float(std)}


def rerank_score(
    text: str,
    sim: float,
    stats: Dict[str, float],
    length_weight: float = 0.05,
    sentence_penalty: float = 0.1,
    digit_weight: float = 0.2,
) -> float:
    cleaned = clean_text(text)
    score = float(sim)

    punct_count = len(_PUNCT_RE.findall(cleaned))
    if punct_count > 1:
        score -= sentence_penalty

    length = len(cleaned)
    mean = float(stats.get("tgt_len_mean", 0.0))
    std = float(stats.get("tgt_len_std", 1.0)) or 1.0
    if mean > 0:
        score -= length_weight * abs(length - mean) / std

    digits = len(_DIGIT_RE.findall(cleaned))
    if length > 0 and digits > 0:
        digit_ratio = digits / length
        score -= digit_weight * digit_ratio

    return score


def select_best(
    candidates: Iterable[Dict[str, Any]],
    stats: Dict[str, float],
    length_weight: float,
    sentence_penalty: float,
    digit_weight: float,
) -> Dict[str, Any]:
    best = None
    best_score = None
    for cand in candidates:
        text = str(cand.get("translation", ""))
        sim = float(cand.get("score", 0.0))
        final = rerank_score(
            text,
            sim,
            stats,
            length_weight=length_weight,
            sentence_penalty=sentence_penalty,
            digit_weight=digit_weight,
        )
        cand = dict(cand)
        cand["final_score"] = final
        if best_score is None or final > best_score:
            best = cand
            best_score = final
    return best or {}
