#!/usr/bin/env python3
"""Dependency-free spectral graph reference kernel for small governed fixtures."""
from __future__ import annotations

from typing import Sequence


def laplacian(adjacency: Sequence[Sequence[float]]) -> list[list[float]]:
    a = [[float(x) for x in row] for row in adjacency]
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("adjacency must be non-empty and square")
    for i in range(n):
        if abs(a[i][i]) > 1e-12:
            raise ValueError("self-loop diagonal must be zero in this reference kernel")
        for j in range(n):
            if abs(a[i][j] - a[j][i]) > 1e-12:
                raise ValueError("reference kernel requires an undirected symmetric graph")
            if a[i][j] < 0:
                raise ValueError("reference kernel requires nonnegative weights")
    degree = [sum(row) for row in a]
    return [[(degree[i] if i == j else 0.0) - a[i][j] for j in range(n)] for i in range(n)]


def dirichlet_energy(signal: Sequence[float], adjacency: Sequence[Sequence[float]]) -> float:
    x = [float(v) for v in signal]
    a = [[float(v) for v in row] for row in adjacency]
    if len(x) != len(a):
        raise ValueError("signal length must equal graph order")
    total = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            total += a[i][j] * (x[i] - x[j]) ** 2
    return total


def graph_receipt(signal: Sequence[float], adjacency: Sequence[Sequence[float]]) -> dict:
    l = laplacian(adjacency)
    return {
        "schema": "gg-math-spectral-graph/v1",
        "laplacian": l,
        "dirichlet_energy": dirichlet_energy(signal, adjacency),
        "interpretation_guard": "Low graph-frequency/smoothness is topology-relative; it does not prove causality, homophily, or truth of an edge.",
        "authority_transfer": False,
    }
