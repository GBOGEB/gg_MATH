#!/usr/bin/env python3
"""Set similarity reference kernel: shingles, Jaccard, MinHash, LSH banding, union-find.

Near-duplicate detection over text (or any set of hashable features) in four
steps, each usable on its own:

1. Features. ``words(text)`` lower-cases and tokenizes; ``shingles(tokens, k)``
   turns the token list into the set of its k-word windows, hashed to 32-bit
   integers with ``zlib.crc32`` over UTF-8 bytes. The hash is fixed across
   processes and Python versions; Python's salted ``hash()`` is never used.
2. Exact similarity. ``jaccard(a, b) = |a & b| / |a | b|``,
   ``overlap(a, b) = |a & b| / min(|a|, |b|)`` (Szymkiewicz-Simpson) and
   ``containment(a, b) = |a & b| / |a|``. Any ratio with an empty denominator
   is 0.0: two empty sets share no evidence of similarity.
3. MinHash. ``minhash(features)`` keeps, for each of ``num_perm`` hash
   functions ``h(x) = (a mix(x) + b) mod p`` with ``p = 2^61 - 1``, the
   minimum over the set. For an ideal random h, P[min h(A) = min h(B)] =
   J(A, B), so the fraction of equal signature positions
   (``estimated_jaccard``) estimates J with standard error
   sqrt(J (1 - J) / num_perm). ``mix`` is the SplitMix64 finalizer applied
   to x mod 2^64; without it a purely linear hash carries arithmetic
   structure in the features (consecutive ids, for example) into the minima
   and biases the estimate. The coefficients come from a seeded SplitMix64
   stream (``minhash_params``), an integer recipe that does not depend on the
   ``random`` module.
4. LSH banding. The signature is cut into ``bands`` bands of
   ``rows = num_perm / bands`` values; two sets become a candidate pair when
   any band matches exactly. A pair with Jaccard s is a candidate with
   probability ``1 - (1 - s^rows)^bands`` (``lsh_probability``), an S-curve
   whose midpoint is near ``(1 / bands)^(1 / rows)`` (``lsh_threshold``).
   Candidates are a filter, not a verdict: callers confirm them with
   ``estimated_jaccard`` or ``jaccard``.

``DisjointSet`` (union-find with path halving and union by size) and
``clusters(pairs)`` turn accepted pairs into connected components.

Which pairs to accept, which thresholds to use and how to label a cluster are
caller policy; this kernel only computes. Dependency-free; runs on the
declared runtime (.python-version).
"""
from __future__ import annotations

import functools
import math
import re
import zlib
from typing import Hashable, Iterable, Sequence

SCHEMA = "gg-math-set-similarity/v1"

MERSENNE_61 = (1 << 61) - 1
"""Prime modulus of the MinHash hash family; every hash value is below it."""
EMPTY_SLOT = MERSENNE_61
"""Signature value of an empty feature set (no real hash value can equal it)."""
DEFAULT_NUM_PERM = 48
DEFAULT_BANDS = 16
DEFAULT_SEED = 7
DEFAULT_SHINGLE = 3
WORD_PATTERN = r"[a-z0-9]+(?:[.,][0-9]+)?"
"""Default token: a run of lower-case letters/digits, keeping one decimal part (``3.5``, ``2,75``)."""

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------- features
def words(text: str | None, pattern: str = WORD_PATTERN) -> list[str]:
    """Tokens of ``text`` after lower-casing, in order (``None`` gives ``[]``)."""
    return re.findall(pattern, (text or "").lower())


def hash_token(token: str) -> int:
    """Deterministic unsigned 32-bit hash of a string: CRC-32 of its UTF-8 bytes."""
    return zlib.crc32(token.encode("utf-8"))


def shingles(tokens: Sequence[str], k: int = DEFAULT_SHINGLE, *, hashed: bool = True) -> set:
    """Set of k-token windows of ``tokens``, each joined by single spaces.

    With ``hashed`` (default) every window is reduced by ``hash_token`` to an
    int; otherwise the joined strings are returned. Fewer than ``k`` tokens
    give the empty set. CRC-32 can map two different windows to one value
    (about 50 % chance of at least one collision among 77 000 windows); use
    ``hashed=False`` when exact set arithmetic matters more than memory.
    """
    if k < 1:
        raise ValueError("shingle size k must be at least 1")
    grams = (" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1))
    return {hash_token(g) for g in grams} if hashed else set(grams)


# ---------------------------------------------------------------- exact similarity
def jaccard(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """|a & b| / |a | b|; 0.0 when both sets are empty."""
    a, b = set(a), set(b)
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def overlap(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """Overlap (Szymkiewicz-Simpson) coefficient |a & b| / min(|a|, |b|); 0.0 if either is empty."""
    a, b = set(a), set(b)
    smaller = min(len(a), len(b))
    return len(a & b) / smaller if smaller else 0.0


def containment(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """Share of ``a`` that is also in ``b``: |a & b| / |a|; 0.0 when ``a`` is empty."""
    a, b = set(a), set(b)
    return len(a & b) / len(a) if a else 0.0


# ---------------------------------------------------------------- MinHash
def _mix64(z: int) -> int:
    """SplitMix64 finalizer: a bijection on 64-bit integers with full avalanche."""
    z &= _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step (Steele, Lea and Flood 2014): returns (new state, output)."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    return state, _mix64(state)


@functools.lru_cache(maxsize=64)
def minhash_params(num_perm: int = DEFAULT_NUM_PERM, seed: int = DEFAULT_SEED) -> tuple[tuple[int, int], ...]:
    """The ``num_perm`` coefficient pairs (a, b) with 1 <= a < p and 0 <= b < p, p = 2^61 - 1.

    Drawn from a SplitMix64 stream started at ``seed`` (two outputs per pair),
    so the same (num_perm, seed) gives the same family on every platform, and
    a longer family extends a shorter one with the same seed.
    """
    if num_perm < 1:
        raise ValueError("num_perm must be at least 1")
    state = seed & _MASK64
    pairs = []
    for _ in range(num_perm):
        state, za = _splitmix64(state)
        state, zb = _splitmix64(state)
        pairs.append((1 + za % (MERSENNE_61 - 1), zb % MERSENNE_61))
    return tuple(pairs)


def minhash(features: Iterable[int], num_perm: int = DEFAULT_NUM_PERM, seed: int = DEFAULT_SEED) -> tuple[int, ...]:
    """MinHash signature of a set of integer features (e.g. ``shingles``).

    Position i holds min over x of (a_i mix(x) + b_i) mod (2^61 - 1), where
    mix is the SplitMix64 finalizer of x mod 2^64 (features that are equal
    mod 2^64 therefore count as one). An empty set gives
    ``(EMPTY_SLOT,) * num_perm``, which ``estimated_jaccard`` and
    ``lsh_candidates`` treat as "no features".
    """
    params = minhash_params(num_perm, seed)
    mixed = {_mix64(int(x)) for x in features}
    if not mixed:
        return (EMPTY_SLOT,) * num_perm
    return tuple(min((a * x + b) % MERSENNE_61 for x in mixed) for a, b in params)


def is_empty_signature(signature: Sequence[int]) -> bool:
    """True for the signature of an empty feature set."""
    return all(v == EMPTY_SLOT for v in signature)


def estimated_jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Fraction of equal positions in two signatures of the same length; 0.0 if either set was empty."""
    if len(sig_a) != len(sig_b) or not sig_a:
        raise ValueError("signatures must be non-empty and of equal length")
    if is_empty_signature(sig_a) or is_empty_signature(sig_b):
        return 0.0
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def minhash_standard_error(similarity: float, num_perm: int) -> float:
    """Standard error sqrt(J (1 - J) / num_perm) of ``estimated_jaccard`` at true Jaccard J."""
    if not 0.0 <= similarity <= 1.0 or num_perm < 1:
        raise ValueError("need 0 <= similarity <= 1 and num_perm >= 1")
    return math.sqrt(similarity * (1.0 - similarity) / num_perm)


# ---------------------------------------------------------------- LSH banding
def _rows(length: int, bands: int) -> int:
    if bands < 1 or length < 1 or length % bands:
        raise ValueError(f"signature length {length} is not a positive multiple of bands={bands}")
    return length // bands


def band_keys(signature: Sequence[int], bands: int = DEFAULT_BANDS) -> list[tuple[int, tuple[int, ...]]]:
    """The ``bands`` bucket keys (band index, band values) of one signature."""
    rows = _rows(len(signature), bands)
    return [(b, tuple(signature[b * rows:(b + 1) * rows])) for b in range(bands)]


def bucket_pairs(groups: Iterable[Iterable[int]], max_bucket: int | None = None) -> set[tuple[int, int]]:
    """All pairs (i, j), i < j, that share a group.

    Groups larger than ``max_bucket`` are skipped: a bucket that large is
    usually shared boilerplate, and it would add a quadratic number of pairs.
    Use it for LSH buckets and for exact-key buckets alike.
    """
    pairs: set[tuple[int, int]] = set()
    for group in groups:
        members = sorted(set(group))
        if len(members) < 2 or (max_bucket is not None and len(members) > max_bucket):
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pairs.add((members[x], members[y]))
    return pairs


def lsh_candidates(signatures: Sequence[Sequence[int]], bands: int = DEFAULT_BANDS,
                   max_bucket: int | None = None) -> set[tuple[int, int]]:
    """Index pairs (i, j), i < j, whose signatures agree on at least one whole band.

    Empty-set signatures never become candidates. ``max_bucket`` is passed to
    ``bucket_pairs``.
    """
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for i, sig in enumerate(signatures):
        if is_empty_signature(sig):
            continue
        for key in band_keys(sig, bands):
            buckets.setdefault(key, []).append(i)
    return bucket_pairs(buckets.values(), max_bucket)


def lsh_probability(similarity: float, bands: int, rows: int) -> float:
    """P[candidate] = 1 - (1 - s^rows)^bands for a pair with Jaccard ``similarity``."""
    if not 0.0 <= similarity <= 1.0 or bands < 1 or rows < 1:
        raise ValueError("need 0 <= similarity <= 1, bands >= 1, rows >= 1")
    return 1.0 - (1.0 - similarity ** rows) ** bands


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate midpoint (1 / bands)^(1 / rows) of the LSH S-curve."""
    if bands < 1 or rows < 1:
        raise ValueError("need bands >= 1 and rows >= 1")
    return (1.0 / bands) ** (1.0 / rows)


# ---------------------------------------------------------------- union-find
class DisjointSet:
    """Union-find over hashable elements, added on first use.

    ``find`` uses path halving and ``union`` links the smaller tree under the
    larger, so a sequence of operations runs in near-linear time.
    """

    def __init__(self, elements: Iterable[Hashable] = ()) -> None:
        self._parent: dict = {}
        self._size: dict = {}
        for e in elements:
            self.add(e)

    def __contains__(self, element: Hashable) -> bool:
        return element in self._parent

    def __len__(self) -> int:
        return len(self._parent)

    def add(self, element: Hashable) -> None:
        if element not in self._parent:
            self._parent[element] = element
            self._size[element] = 1

    def find(self, element: Hashable) -> Hashable:
        """Representative of ``element``'s set (adds it as a singleton if new)."""
        self.add(element)
        parent = self._parent
        while parent[element] != element:
            parent[element] = parent[parent[element]]
            element = parent[element]
        return element

    def union(self, a: Hashable, b: Hashable) -> bool:
        """Merge the sets of ``a`` and ``b``; True when they were separate."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        return True

    def connected(self, a: Hashable, b: Hashable) -> bool:
        return self.find(a) == self.find(b)

    def components(self, min_size: int = 1) -> list[list]:
        """Sets with at least ``min_size`` members, each sorted, ordered by first member.

        Elements must be mutually orderable (all ints, or all strings, ...).
        """
        groups: dict = {}
        for e in self._parent:
            groups.setdefault(self.find(e), []).append(e)
        return sorted((sorted(g) for g in groups.values() if len(g) >= min_size), key=lambda g: g[0])


def clusters(pairs: Iterable[tuple[Hashable, Hashable]], nodes: Iterable[Hashable] = ()) -> list[list]:
    """Connected components of the graph whose edges are ``pairs``.

    Without ``nodes`` only elements that occur in a pair are returned (every
    component has at least two members, unless a pair links an element to
    itself). Elements listed in ``nodes`` but in no pair appear as singletons.
    """
    ds = DisjointSet(nodes)
    for a, b in pairs:
        ds.union(a, b)
    return ds.components()


__all__ = [
    "SCHEMA", "MERSENNE_61", "EMPTY_SLOT", "DEFAULT_NUM_PERM", "DEFAULT_BANDS", "DEFAULT_SEED", "DEFAULT_SHINGLE",
    "WORD_PATTERN", "words", "hash_token", "shingles", "jaccard", "overlap", "containment", "minhash_params",
    "minhash", "is_empty_signature", "estimated_jaccard", "minhash_standard_error", "band_keys", "bucket_pairs",
    "lsh_candidates", "lsh_probability", "lsh_threshold", "DisjointSet", "clusters",
]
