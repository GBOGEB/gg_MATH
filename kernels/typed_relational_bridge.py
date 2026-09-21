#!/usr/bin/env python3
"""Typed LM-10 H05 bridge across Pearson, Bradley-Terry and PCA diagnostics.

The bridge deliberately preserves evidence semantics instead of converting one
statistical object into another. Pearson correlation remains paired continuous
association with uncertainty; Bradley-Terry remains observed pairwise outcomes;
PCA remains multivariate geometry. PCA scores/loadings may annotate or stratify
an already-governed comparison population, but they do not manufacture BT wins,
causality, engineering authority or project acceptance.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from kernels.bradley_terry import rank_pairs
from kernels.stats_core import fisher_z_interval, pearson_correlation


def pearson_with_uncertainty(
    x: Sequence[float],
    y: Sequence[float],
    *,
    z_critical: float = 1.959963984540054,
) -> dict:
    if len(x) != len(y):
        raise ValueError("length mismatch")
    r = pearson_correlation(x, y)
    lo, hi = fisher_z_interval(r, len(x), z_critical=z_critical)
    return {
        "kind": "PEARSON_ASSOCIATION",
        "n": len(x),
        "r": r,
        "fisher_z_interval": [lo, hi],
        "uncertainty_method": "FISHER_Z_APPROXIMATION",
        "causality_claimed": False,
        "authority_transfer": False,
    }


def bradley_terry_observed_only(pairs: Iterable[Sequence[str]], *, iterations: int = 64) -> dict:
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each pair must be [winner, loser]")
        winner, loser = str(pair[0]), str(pair[1])
        normalized.append((winner, loser))
    result = rank_pairs(normalized, iterations=iterations)
    return {
        "kind": "BRADLEY_TERRY_OBSERVED_PAIRWISE",
        "input_pair_count": len(normalized),
        "observed_outcomes_only": True,
        "result": result,
        "pca_generated_wins": False,
        "authority_transfer": False,
    }


def typed_bt_pca_bridge(
    *,
    observed_pairs: Iterable[Sequence[str]],
    pca_annotations: dict[str, dict] | None = None,
    population_id: str,
    evidence_class: str,
) -> dict:
    """Bind BT and PCA diagnostics without semantic conversion.

    ``pca_annotations`` can carry already-computed descriptive fields (scores,
    loadings, component labels, subspace metrics) keyed by item. It is never
    converted into pairwise outcomes. Empty observed comparisons therefore
    remain a Bradley-Terry DEFER even when PCA annotations are present.
    """
    if not population_id:
        raise ValueError("population_id is required")
    if not evidence_class:
        raise ValueError("evidence_class is required")

    bt = bradley_terry_observed_only(observed_pairs)
    annotations = dict(pca_annotations or {})
    result = bt["result"]

    return {
        "schema": "gg-math-lm10-h05-typed-relational-bridge/v1",
        "population_id": population_id,
        "evidence_class": evidence_class,
        "bt": bt,
        "pca": {
            "kind": "DESCRIPTIVE_MULTIVARIATE_ANNOTATION",
            "annotations": annotations,
            "annotation_count": len(annotations),
            "may_generate_pairwise_outcomes": False,
            "may_define_causality": False,
        },
        "consumer_disposition": (
            "DEFER_BT_NO_VALID_CONNECTED_OBSERVED_GRAPH"
            if result["status"].startswith("DEFER_")
            else "REFERENCE_RELATIONAL_DIAGNOSTIC_AVAILABLE"
        ),
        "semantic_guards": {
            "pearson_is_not_causality": True,
            "pca_is_not_bradley_terry_evidence": True,
            "pca_loading_sign_requires_alignment_before_directional_interpretation": True,
            "bt_requires_observed_pairwise_outcomes": True,
            "disconnected_bt_graph_defers": True,
            "hard_gate_compensation_allowed": False,
            "authority_transfer": False,
        },
    }
