"""Unit checks for the dominance relation and the non-dominated sorter.

Mirrors the Week 4 lab checks: four dominance assertions (including the
equal-vector case) and two invariants over 200 random objective vectors.
Run with:  python code/test_dominance.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizers import _dominates, nondominated_sort  # noqa: E402


def test_dominance_pairs():
    # Better on both objectives.
    assert _dominates(np.array([3.0, 8.0]), np.array([5.0, 9.0]))
    # Equal on the first, strictly better on the second.
    assert _dominates(np.array([3.0, 8.0]), np.array([3.0, 9.0]))
    # Each wins one objective: incomparable.
    assert not _dominates(np.array([3.0, 8.0]), np.array([5.0, 6.0]))
    # Identical vectors: "strictly better on one" fails.
    assert not _dominates(np.array([3.0, 8.0]), np.array([3.0, 8.0]))


def test_sort_invariants():
    rng = np.random.default_rng(20260909)
    objs = [tuple(v) for v in rng.random((200, 2))]
    fronts = nondominated_sort(objs)

    seen = [i for front in fronts for i in front]
    assert sorted(seen) == list(range(200)), "every index must appear exactly once"
    assert len(seen) == len(set(seen)), "no index may appear in two fronts"

    for r, front in enumerate(fronts[:-1]):
        for p in front:
            for later in fronts[r + 1:]:
                for q in later:
                    assert not _dominates(np.array(objs[q]), np.array(objs[p])), (
                        "a point in a later front dominates an earlier-front point"
                    )


if __name__ == "__main__":
    test_dominance_pairs()
    test_sort_invariants()
    print("dominance: 4 assertions passed")
    print("sort invariants over 200 random points: passed")
