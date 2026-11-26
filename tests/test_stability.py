"""Tests for segmentation.stability (Phase 3.4)."""

from __future__ import annotations

import pytest
from pyspark.ml.feature import VectorAssembler

from segmentation.stability import (
    StabilityResult,
    adjusted_rand_index,
    ari_between,
    assign_nearest,
    choose_final,
    fit_and_label,
    stability,
)

Z = ["z_a", "z_b"]


@pytest.fixture(scope="module")
def blobs(spark):
    rows, cid = [], 0
    offsets = [(-0.3, 0.1), (0.2, -0.2), (0.1, 0.3), (-0.1, -0.1), (0.3, 0.2), (0.0, -0.3)]
    for cx, cy in [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]:
        for dx, dy in offsets:
            rows.append((cid, cx + dx, cy + dy))
            cid += 1
    df = spark.createDataFrame(rows, "customer_id INT, z_a DOUBLE, z_b DOUBLE")
    return VectorAssembler(inputCols=Z, outputCol="features").transform(df)


# ----------------------------------------------------------------------------- ARI


def test_ari_identical_and_relabelled_partitions_are_1():
    a = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert adjusted_rand_index(a, [5, 5, 3, 3, 9, 9]) == pytest.approx(1.0)


def test_ari_known_value():
    # scikit-learn docs example: adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2]) = 0.5714...
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)


def test_ari_unrelated_partition_is_not_positive():
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) <= 0


def test_ari_edge_cases():
    assert adjusted_rand_index([0], [1]) == 1.0
    assert adjusted_rand_index([0, 0, 0], [1, 1, 1]) == 1.0
    with pytest.raises(ValueError):
        adjusted_rand_index([0, 1], [0])


def test_ari_between_aligns_on_ids():
    x = {1: 0, 2: 0, 3: 1}
    y = {3: 7, 1: 4, 2: 4}  # same partition, different order and names
    assert ari_between(x, y) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ari_between(x, {1: 0, 2: 0})


# ----------------------------------------------------------------------------- assignment


def test_assign_nearest_matches_kmeans_rule(spark):
    df = spark.createDataFrame(
        [(1, 0.0, 0.0), (2, 9.0, 1.0), (3, 5.0, 0.0)], "id INT, z_a DOUBLE, z_b DOUBLE"
    )
    centers = [[0.0, 0.0], [10.0, 0.0]]
    labels = {r["id"]: r["cluster"] for r in assign_nearest(df, centers, Z).collect()}
    assert labels == {1: 0, 2: 1, 3: 0}  # id 3 is a tie -> lowest index
    one = {r["id"]: r["cluster"] for r in assign_nearest(df, [[1.0, 1.0]], Z).collect()}
    assert set(one.values()) == {0}
    assert assign_nearest(df, centers, Z).columns == ["id", "z_a", "z_b", "cluster"]


@pytest.mark.parametrize("algorithm", ["kmeans", "bisecting_kmeans", "gmm"])
def test_fit_and_label_labels_every_customer(blobs, algorithm):
    # fit on half, label everyone (GMM needs more than 3 points per 2-D component: all rows)
    sample = blobs if algorithm == "gmm" else blobs.where("customer_id % 2 = 0")
    labels = fit_and_label(algorithm, sample, blobs, 3, 1, Z)
    assert sorted(labels) == list(range(18))
    assert len(set(labels.values())) == 3


def test_fit_and_label_unknown_algorithm(blobs):
    with pytest.raises(ValueError):
        fit_and_label("dbscan", blobs, blobs, 3, 1, Z)


# ----------------------------------------------------------------------------- stability


def test_stability_on_clear_clusters_is_perfect(blobs):
    r = stability("kmeans", blobs, 3, Z, seeds=(1, 2, 3), bootstrap_seeds=(11, 12))
    assert r.error is None
    assert len(r.seed_ari) == 3  # pairs of 3 seeds
    assert len(r.bootstrap_ari) == 2
    s = r.summary()
    assert s["seed_ari_mean"] == pytest.approx(1.0)
    assert s["bootstrap_ari_min"] == pytest.approx(1.0)
    assert r.to_dict()["reference_seed"] == 1


def test_stability_records_errors(blobs):
    r = stability("dbscan", blobs, 3, Z, seeds=(1, 2), bootstrap_seeds=(3,))
    assert r.error.startswith("ValueError")
    assert r.summary()["seed_ari_mean"] is None


def test_summary_of_empty_result():
    assert StabilityResult("kmeans", 3, 1, [1], [2]).summary() == {
        "seed_ari_mean": None,
        "seed_ari_min": None,
        "bootstrap_ari_mean": None,
        "bootstrap_ari_min": None,
    }


# ----------------------------------------------------------------------------- final choice


def _sweep(algo, k, sil, min_share=0.1):
    return {"algorithm": algo, "k": k, "silhouette": sil, "min_share": min_share, "error": None}


def _stab(algo, k, seed_ari, boot_ari, error=None):
    return {
        "algorithm": algo,
        "k": k,
        "seed_ari_mean": seed_ari,
        "bootstrap_ari_mean": boot_ari,
        "error": error,
    }


def test_choose_final_prefers_larger_k_within_tolerance():
    sweep = [_sweep("kmeans", 3, 0.474), _sweep("kmeans", 4, 0.468), _sweep("kmeans", 5, 0.40)]
    stab = [
        _stab("kmeans", 3, 0.99, 0.95),
        _stab("kmeans", 4, 0.97, 0.9),
        _stab("kmeans", 5, 0.9, 0.9),
    ]
    pick = choose_final(sweep, stab)
    assert (pick["algorithm"], pick["k"]) == ("kmeans", 4)
    assert pick["best_silhouette"] == 0.474
    assert pick["eligible"] == [("kmeans", 3), ("kmeans", 4), ("kmeans", 5)]


def test_choose_final_drops_unstable_small_and_failed():
    sweep = [
        _sweep("kmeans", 3, 0.47),
        _sweep("kmeans", 4, 0.47),  # unstable
        _sweep("kmeans", 5, 0.48, min_share=0.01),  # tiny segment
        _sweep("gmm", 3, 0.49),  # failed stability run
    ]
    stab = [
        _stab("kmeans", 3, 0.95, 0.9),
        _stab("kmeans", 4, 0.6, 0.9),
        _stab("kmeans", 5, 0.95, 0.95),
        _stab("gmm", 3, None, None, error="boom"),
    ]
    pick = choose_final(sweep, stab)
    assert (pick["algorithm"], pick["k"]) == ("kmeans", 3)
    assert choose_final(sweep, [_stab("kmeans", 4, 0.6, 0.9)]) is None


def test_choose_final_separate_bootstrap_threshold():
    """Phase 4: seed ARI must still be >= 0.8, bootstrap only >= 0.7."""
    sweep = [
        _sweep("bisecting_kmeans", 3, 0.364),
        _sweep("kmeans", 3, 0.354),
        _sweep("kmeans", 5, 0.40),
    ]
    stab = [
        _stab("bisecting_kmeans", 3, 0.997, 0.744),
        _stab("kmeans", 3, 0.981, 0.743),
        _stab("kmeans", 5, 0.62, 0.75),  # seeds unstable: still out
    ]
    assert choose_final(sweep, stab) is None  # the original single 0.8 rule
    pick = choose_final(sweep, stab, min_bootstrap_ari=0.7)
    assert (pick["algorithm"], pick["k"]) == ("bisecting_kmeans", 3)
    assert pick["eligible"] == [("bisecting_kmeans", 3), ("kmeans", 3)]
    assert (pick["min_seed_ari"], pick["min_bootstrap_ari"]) == (0.8, 0.7)
