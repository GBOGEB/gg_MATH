#!/usr/bin/env python3
"""Seed tree: one root seed, one independent random stream per run cell.

A statistical run is a grid of cells indexed along named axes, by default
``(epoch, set, replicate, duplicate)``. Every node of that grid, a full cell
or a prefix such as ``(epoch, set, replicate)``, gets its own NumPy
``SeedSequence`` with ``entropy=root_seed`` and ``spawn_key=index``. This is
exactly the node that nested ``SeedSequence(root).spawn(...)`` calls reach,
built directly, so:

* Random access. Any node regenerates on its own, without replaying the
  nodes before it.
* Stable growth. Adding an epoch, set, replicate or duplicate adds new nodes
  and never shifts the stream of an existing one. No axis has a declared
  size.
* Independence. Distinct indices give distinct ``SeedSequence`` states, which
  NumPy's hashing makes statistically independent for practical purposes
  (not a cryptographic guarantee).
* Replay. ``receipt`` records the root seed, the axes, the index set and a
  SHA-256 over every node's state fingerprint; ``replay`` re-derives and
  compares them.

Streams use ``PCG64``. NumPy keeps ``SeedSequence`` and the raw ``PCG64``
bit stream stable across releases, and ``Generator.random()`` maps it to
doubles. Other ``Generator`` distributions may change between NumPy
releases, so a run that must replay bit-for-bit draws uniforms and
transforms them itself (see ``research/seed_tree_dsm_makespan.py``).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA = "gg-math-seed-tree/v1"
DEFAULT_AXES = ("epoch", "set", "replicate", "duplicate")
BIT_GENERATOR = "PCG64"
DERIVATION = "SeedSequence(entropy=root_seed, spawn_key=index)"


def _index_part(value: object, axis: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{axis} index must be an integer >= 0, got {value!r}")
    return int(value)


class SeedTree:
    """Deterministic map from a grid index to an independent random stream."""

    def __init__(self, root_seed: int, axes: Sequence[str] = DEFAULT_AXES) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
            raise ValueError("root_seed must be an integer >= 0")
        axes = tuple(axes)
        if not axes or any(not isinstance(a, str) or not a for a in axes):
            raise ValueError("axes must be non-empty names")
        if len(set(axes)) != len(axes):
            raise ValueError("axis names must be unique")
        self.root_seed = root_seed
        self.axes = axes

    def index(self, *index: object) -> tuple[int, ...]:
        """Validated node index: a prefix of the axes, from the root () to a full cell."""
        if len(index) > len(self.axes):
            raise ValueError(f"index has {len(index)} parts; the tree has {len(self.axes)} axes")
        return tuple(_index_part(v, a) for v, a in zip(index, self.axes))

    def seed_sequence(self, *index: object) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.root_seed, spawn_key=self.index(*index))

    def rng(self, *index: object) -> np.random.Generator:
        """A fresh ``Generator`` for the node; the same index always gives the same stream."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(*index)))

    def fingerprint(self, *index: object) -> str:
        """128-bit hex of the node's SeedSequence state; equal iff the streams are equal."""
        words = self.seed_sequence(*index).generate_state(4, np.uint32)
        return "".join(f"{int(w):08x}" for w in words)

    def receipt(self, nodes: Iterable[Sequence[int]], *, include_nodes: bool = False) -> dict:
        """Everything needed to replay the run's streams and check them."""
        idx = [self.index(*n) for n in nodes]
        if len(set(idx)) != len(idx):
            raise ValueError("duplicate node index in receipt")
        prints = [self.fingerprint(*n) for n in idx]
        out = {
            "schema": SCHEMA,
            "root_seed": self.root_seed,
            "axes": list(self.axes),
            "bit_generator": BIT_GENERATOR,
            "derivation": DERIVATION,
            "n_nodes": len(idx),
            "nodes_sha256": _digest(idx),
            "fingerprints_sha256": _digest(prints),
            "distinct_fingerprints": len(set(prints)) == len(prints),
        }
        if include_nodes:
            out["nodes"] = [{"index": list(n), "fingerprint": p} for n, p in zip(idx, prints)]
        return out


def grid(*sizes: int) -> Iterator[tuple[int, ...]]:
    """All indices of a full grid in row-major order, e.g. ``grid(E, S, R, D)``."""
    return itertools.product(*(range(s) for s in sizes))


def nodes_for_grid(*sizes: int) -> list[tuple[int, ...]]:
    """Every node of a grid at every depth: the cells and all their prefixes (root excluded)."""
    return [n for depth in range(1, len(sizes) + 1) for n in grid(*sizes[:depth])]


def replay(receipt: Mapping, nodes: Iterable[Sequence[int]]) -> dict:
    """Re-derive a receipt from its root seed and axes and report any field that differs."""
    if receipt.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} receipt")
    if receipt.get("bit_generator") != BIT_GENERATOR or receipt.get("derivation") != DERIVATION:
        raise ValueError("receipt uses a different generator or derivation")
    tree = SeedTree(receipt["root_seed"], receipt["axes"])
    again = tree.receipt(nodes, include_nodes="nodes" in receipt)
    mismatches = sorted(k for k in again if again[k] != receipt.get(k))
    return {"ok": not mismatches, "mismatches": mismatches}


def _digest(items: Sequence) -> str:
    text = json.dumps([list(i) if isinstance(i, tuple) else i for i in items], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
