#!/usr/bin/env python3
"""Known-answer challenges for the set similarity kernel. Runs under pytest or as a plain script."""
from __future__ import annotations

import math
import statistics
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import set_similarity as ss


def _raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def _doc(tag: str, n: int = 30) -> list[str]:
    """A synthetic document of ``n`` distinct tokens; documents with different tags share none."""
    return [tag + "w" + str(j) for j in range(n)]


def test_words_keep_one_decimal_part():
    assert ss.words("Pump P-101, 3.5 bar; 2,75 m") == ["pump", "p", "101", "3.5", "bar", "2,75", "m"]
    assert ss.words(None) == [] and ss.words("") == []


def test_shingles_known_crc32():
    toks = ["a", "b", "c", "d"]
    assert ss.shingles(toks, 3, hashed=False) == {"a b c", "b c d"}
    # CRC-32 is fixed by the zlib/PNG standard, so these literals never move.
    assert ss.shingles(toks, 3) == {3358461392, 271091860} == {zlib.crc32(b"a b c"), zlib.crc32(b"b c d")}
    assert ss.shingles(toks, 1, hashed=False) == {"a", "b", "c", "d"}
    assert ss.shingles(["a", "b"], 3) == set()
    assert _raises(ss.shingles, toks, 0)


def test_jaccard_overlap_containment_fixed_sets():
    a, b = {1, 2, 3}, {2, 3, 4}
    assert ss.jaccard(a, b) == 0.5
    assert math.isclose(ss.overlap(a, b), 2 / 3)
    assert math.isclose(ss.containment(a, b), 2 / 3)
    small, big = {1, 2}, {1, 2, 3, 4, 5}
    assert math.isclose(ss.jaccard(small, big), 0.4)
    assert ss.overlap(small, big) == 1.0 and ss.containment(small, big) == 1.0
    assert ss.containment(big, small) == 0.4
    assert ss.jaccard(a, a) == 1.0 and ss.jaccard(a, {9}) == 0.0
    # Empty denominators give 0.0, never a ZeroDivisionError.
    assert ss.jaccard(set(), set()) == 0.0 and ss.overlap(set(), a) == 0.0 and ss.containment(set(), a) == 0.0


def test_minhash_params_are_pinned_and_prefix_stable():
    # The SplitMix64 recipe is part of the contract: a change here changes every signature.
    assert ss.minhash_params(2, 7) == ((273560573251292638, 309689372594955804),
                                       (475200682319751697, 1529793891446696399))
    long, short = ss.minhash_params(64, 3), ss.minhash_params(16, 3)
    assert long[:16] == short
    assert all(1 <= a < ss.MERSENNE_61 and 0 <= b < ss.MERSENNE_61 for a, b in long)
    assert ss.minhash_params(16, 4) != short
    assert _raises(ss.minhash_params, 0)


def test_minhash_identical_disjoint_and_empty():
    a = ss.shingles(_doc("x"), 3)
    sig = ss.minhash(a)
    assert len(sig) == ss.DEFAULT_NUM_PERM and sig == ss.minhash(set(a))
    assert ss.estimated_jaccard(sig, ss.minhash(a)) == 1.0
    assert ss.estimated_jaccard(sig, ss.minhash(ss.shingles(_doc("y"), 3))) == 0.0
    empty = ss.minhash(set())
    assert ss.is_empty_signature(empty) and not ss.is_empty_signature(sig)
    assert ss.estimated_jaccard(empty, empty) == 0.0 and ss.estimated_jaccard(empty, sig) == 0.0
    assert _raises(ss.estimated_jaccard, sig, sig[:-1])


def test_minhash_estimate_within_three_standard_errors():
    a = {ss.hash_token("t" + str(i)) for i in range(300)}
    b = {ss.hash_token("t" + str(i)) for i in range(100, 400)}
    truth = ss.jaccard(a, b)
    assert truth == 0.5
    for num_perm in (48, 128, 256):
        est = ss.estimated_jaccard(ss.minhash(a, num_perm), ss.minhash(b, num_perm))
        assert abs(est - truth) <= 3 * ss.minhash_standard_error(truth, num_perm), (num_perm, est)
    assert math.isclose(ss.minhash_standard_error(0.5, 256), 1 / 32)


def test_minhash_unbiased_on_consecutive_integers():
    # A purely linear (a x + b) mod p hash gives a mean near 0.44 here; the SplitMix64 pre-mix removes the bias.
    a, b = set(range(300)), set(range(100, 400))
    seeds = range(32)
    ests = [ss.estimated_jaccard(ss.minhash(a, 128, s), ss.minhash(b, 128, s)) for s in seeds]
    se_mean = ss.minhash_standard_error(0.5, 128) / math.sqrt(len(seeds))
    assert abs(statistics.mean(ests) - 0.5) <= 3 * se_mean, statistics.mean(ests)


def test_lsh_curve():
    assert math.isclose(ss.lsh_threshold(16, 3), (1 / 16) ** (1 / 3))
    assert math.isclose(ss.lsh_probability(0.5, 16, 3), 1 - (1 - 0.125) ** 16)
    assert ss.lsh_probability(0.0, 16, 3) == 0.0 and ss.lsh_probability(1.0, 16, 3) == 1.0
    assert ss.lsh_probability(0.3, 16, 3) < ss.lsh_probability(0.6, 16, 3)
    assert _raises(ss.lsh_probability, 1.5, 16, 3) and _raises(ss.lsh_threshold, 0, 3)


def test_band_keys_split_the_signature():
    sig = tuple(range(48))
    keys = ss.band_keys(sig, 16)
    assert len(keys) == 16 and keys[0] == (0, (0, 1, 2)) and keys[15] == (15, (45, 46, 47))
    assert _raises(ss.band_keys, sig, 5)


def test_bucket_pairs_and_max_bucket():
    assert ss.bucket_pairs([[3, 1, 2], [4], [5, 5]]) == {(1, 2), (1, 3), (2, 3)}
    assert ss.bucket_pairs([[1, 2, 3], [7, 8]], max_bucket=2) == {(7, 8)}


def test_lsh_finds_planted_near_duplicate():
    docs = [_doc("d" + str(i) + "_") for i in range(40)]
    planted = list(docs[7])
    planted[-1] = "changed"
    docs.append(planted)  # index 40: same as doc 7 except the last token
    sets = [ss.shingles(d, 3) for d in docs]
    assert math.isclose(ss.jaccard(sets[7], sets[40]), 27 / 29)
    sigs = [ss.minhash(s) for s in sets] + [ss.minhash(set())]
    cand = ss.lsh_candidates(sigs, bands=16)
    assert cand == {(7, 40)}
    assert ss.estimated_jaccard(sigs[7], sigs[40]) >= 0.75


def test_disjoint_set_and_clusters():
    ds = ss.DisjointSet(range(3))
    assert ds.union(0, 1) and not ds.union(1, 0)
    assert ds.connected(0, 1) and not ds.connected(0, 2)
    assert len(ds) == 3 and 2 in ds and 9 not in ds
    assert ds.components() == [[0, 1], [2]] and ds.components(min_size=2) == [[0, 1]]
    pairs = [(1, 2), (2, 3), (5, 4), (6, 6)]
    assert ss.clusters(pairs) == [[1, 2, 3], [4, 5], [6]]
    assert ss.clusters(pairs, nodes=range(8)) == [[0], [1, 2, 3], [4, 5], [6], [7]]
    assert ss.clusters([("b", "a"), ("c", "d"), ("d", "b")]) == [["a", "b", "c", "d"]]
    # A long chain stays correct (path halving, union by size).
    chain = ss.DisjointSet()
    for i in range(1000):
        chain.union(i, i + 1)
    assert chain.components() == [list(range(1001))]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
