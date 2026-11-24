"""Tests for segmentation.metrics, segmentation.bisecting, segmentation.clustering and
segmentation.tracking (Phase 3.3). Uses three well-separated 2-D blobs, so the right answer
(k=3, pure clusters) is known."""

from __future__ import annotations

import pytest
from pyspark.ml.feature import VectorAssembler

import segmentation.clustering as clustering
from segmentation.bisecting import Split, assign, fit_bisecting
from segmentation.clustering import TrialResult, run_sweep, run_trial
from segmentation.metrics import (
    cluster_stats,
    elbow_k,
    silhouette,
    size_shares,
    suggest_k,
    wcss,
)
from segmentation.tracking import flat_row, results_table, trial_run_name

Z = ["z_a", "z_b"]
CENTRES = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (0.0, 10.0)}


@pytest.fixture(scope="module")
def blobs(spark):
    rows, cid = [], 0
    offsets = [(-0.3, 0.1), (0.2, -0.2), (0.1, 0.3), (-0.1, -0.1), (0.3, 0.2), (0.0, -0.3)]
    for blob, (cx, cy) in CENTRES.items():
        for dx, dy in offsets:
            rows.append((cid, blob, cx + dx, cy + dy))
            cid += 1
    df = spark.createDataFrame(rows, "customer_id INT, true_blob INT, z_a DOUBLE, z_b DOUBLE")
    return VectorAssembler(inputCols=Z, outputCol="features").transform(df)


def _purity(df, label_col="cluster"):
    """Each predicted cluster contains exactly one true blob and vice versa."""
    pairs = {(r["true_blob"], r[label_col]) for r in df.select("true_blob", label_col).collect()}
    return len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs})


# ----------------------------------------------------------------------------- metrics


def test_cluster_stats_centroid_and_sse(spark):
    df = spark.createDataFrame(
        [(0, 0.0, 0.0), (0, 2.0, 0.0), (1, 10.0, 10.0)], "cluster INT, z_a DOUBLE, z_b DOUBLE"
    )
    stats = cluster_stats(df, "cluster", Z)
    assert stats[0]["n"] == 2
    assert stats[0]["centroid"] == pytest.approx([1.0, 0.0])
    assert stats[0]["sse"] == pytest.approx(2.0)  # (1^2 + 1^2)
    assert stats[1]["sse"] == pytest.approx(0.0)
    assert wcss(stats) == pytest.approx(2.0)
    assert size_shares(stats, 3) == pytest.approx({0: 2 / 3, 1: 1 / 3, 2: 0.0})


def test_silhouette_high_for_true_labels_none_for_one_cluster(blobs):
    assert silhouette(blobs.withColumnRenamed("true_blob", "cluster"), "features", "cluster") > 0.9
    one = blobs.withColumn("cluster", blobs["true_blob"] * 0)
    assert silhouette(one, "features", "cluster") is None


def test_elbow_k():
    assert elbow_k([2, 3, 4, 5, 6], [100, 30, 20, 15, 12]) == 3
    assert elbow_k([2, 3, 4, 5, 6], [100, 80, 60, 40, 20]) == 2  # straight line: no elbow
    assert elbow_k([2, 3], [10, 5]) == 2


def _r(algo, k, sil, min_share, error=None):
    return {
        "algorithm": algo,
        "k": k,
        "error": error,
        "metrics": {"silhouette": sil, "min_share": min_share},
    }


def test_suggest_k_rules():
    results = [
        _r("kmeans", 2, 0.90, 0.40),  # excluded: k < k_min
        _r("kmeans", 3, 0.60, 0.10),
        _r("kmeans", 4, 0.70, 0.01),  # excluded: tiny segment
        _r("kmeans", 5, 0.60, 0.05),  # ties with k=3 -> smaller k wins
        _r("kmeans", 6, 0.99, 0.20, error="boom"),  # failed run
        _r("gmm", 3, 0.50, 0.10),
    ]
    assert suggest_k(results, "kmeans") == 3
    assert suggest_k(results, "kmeans", min_share=0.0) == 4
    assert suggest_k(results, "kmeans", k_min=2) == 2
    assert suggest_k(results, "bisecting_kmeans") is None


# ----------------------------------------------------------------------------- bisecting


def test_fit_bisecting_recovers_blobs_and_truncates(blobs):
    splits = fit_bisecting(blobs, 3, Z, seed=1)
    assert [(s.parent, s.child) for s in splits][0] == (0, 1)
    assert len(splits) == 2
    assert splits[0].parent_size == 18
    k3 = assign(blobs, splits, Z, k=3)
    assert sorted(r["cluster"] for r in k3.select("cluster").distinct().collect()) == [0, 1, 2]
    assert _purity(k3)
    k2 = assign(blobs, splits, Z, k=2)  # first split only
    assert {r["cluster"] for r in k2.select("cluster").distinct().collect()} == {0, 1}


def test_assign_ties_go_left_and_other_leaves_untouched(spark):
    df = spark.createDataFrame([(0.0, 0.0), (5.0, 0.0), (9.0, 0.0)], "z_a DOUBLE, z_b DOUBLE")
    s1 = Split(0, 1, (0.0, 0.0), (10.0, 0.0), 3, 1.0)  # x=5 is a tie -> stays 0
    s2 = Split(1, 2, (8.0, 0.0), (9.0, 0.0), 1, 0.0)  # only rows already in leaf 1 move
    labels = [r["cluster"] for r in assign(df, [s1, s2], Z).collect()]
    assert labels == [0, 0, 2]


def test_split_dict_round_trip():
    s = Split(0, 1, (0.0, 1.0), (2.0, 3.0), 10, 4.5)
    assert Split.from_dict(s.to_dict()) == s


# ----------------------------------------------------------------------------- trials


@pytest.mark.parametrize(
    ("algorithm", "impl"),
    [
        ("kmeans", "spark_ml"),
        ("gmm", "spark_ml"),
        ("bisecting_kmeans", "kmeans_splits"),
        ("bisecting_kmeans", "spark_ml"),
    ],
)
def test_run_trial_finds_the_three_blobs(blobs, algorithm, impl):
    r = run_trial(algorithm, blobs, 3, Z, seed=1, bisecting_impl=impl)
    assert r.error is None, r.error
    assert r.implementation == impl
    assert r.metrics["silhouette"] > 0.9
    assert r.metrics["min_share"] == pytest.approx(1 / 3)
    assert r.metrics["n_empty_clusters"] == 0
    assert r.metrics["fit_eval_seconds"] >= 0
    assert sum(r.shares.values()) == pytest.approx(1.0)
    if algorithm == "gmm":
        assert r.metrics["mean_max_probability"] > 0.95
        assert len(r.model_info["means"]) == 3
    elif impl == "kmeans_splits":
        assert len(r.model_info["splits"]) == 2
    else:
        assert len(r.model_info["centers"]) == 3


def test_run_trial_records_errors_instead_of_raising(blobs):
    r = run_trial("dbscan", blobs, 3, Z)
    assert r.error.startswith("ValueError")
    assert "fit_eval_seconds" in r.metrics


def test_run_sweep_fits_bisecting_tree_once(blobs, monkeypatch):
    calls = []
    real = clustering.fit_bisecting

    def counting(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(clustering, "fit_bisecting", counting)
    seen = []
    results = run_sweep(blobs, Z, [2, 3, 4], seed=1, on_result=seen.append)
    assert calls == [4]  # one tree, up to max k
    assert len(results) == len(seen) == 9
    assert [(r.algorithm, r.k) for r in results][:3] == [
        ("kmeans", 2),
        ("kmeans", 3),
        ("kmeans", 4),
    ]
    best = {
        a: max((r for r in results if r.algorithm == a), key=lambda r: r.metrics["silhouette"]).k
        for a in clustering.ALGORITHMS
    }
    assert best == {"kmeans": 3, "bisecting_kmeans": 3, "gmm": 3}
    wc = {r.k: r.metrics["wcss"] for r in results if r.algorithm == "kmeans"}
    assert wc[2] > wc[3] >= wc[4]


def test_run_sweep_rejects_unknown_bisecting_impl(blobs):
    with pytest.raises(ValueError):
        run_sweep(blobs, Z, [2], bisecting_impl="sklearn")


# ----------------------------------------------------------------------------- tracking


def test_flat_rows_and_names():
    r = TrialResult("kmeans", 4, 42, "spark_ml", metrics={"silhouette": 0.5, "wcss": 10.0})
    row = flat_row(r)
    assert row == {
        "algorithm": "kmeans",
        "k": 4,
        "implementation": "spark_ml",
        "error": None,
        "silhouette": 0.5,
        "wcss": 10.0,
    }
    assert results_table([r, r.to_dict()]) == [row, row]
    assert trial_run_name(r) == "kmeans_k4"
