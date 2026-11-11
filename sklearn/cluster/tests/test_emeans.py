import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sklearn.cluster import EMeans, KMeans
from sklearn.cluster._emeans import (
    _epochs_n_rows,
    _resolve_batch_size,
    _resolve_initial_step_size,
    _revive_dead_clusters,
)
from sklearn.cluster._k_means_emeans import _emeans_step_dense, _emeans_step_sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS


def _reference_step(
    X, sample_weight, batch_indices, centers, weight_ema, step_size, epoch_stats
):
    """NumPy implementation of one e-means step, updating its inputs inplace."""
    X_batch = X[batch_indices]
    if hasattr(X_batch, "toarray"):
        X_batch = X_batch.toarray()
    sw_batch = sample_weight[batch_indices]
    distances = ((X_batch[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    inertia = (sw_batch * distances[np.arange(len(labels)), labels]).sum()

    for j in range(centers.shape[0]):
        mask = labels == j
        weight = sw_batch[mask].sum()
        retained = (1 - step_size) * weight_ema[j]
        if weight <= 0:
            weight_ema[j] = retained
            continue
        sums = (sw_batch[mask, None] * X_batch[mask]).sum(axis=0)
        if epoch_stats is not None:
            epoch_stats[0][j] += sums
            epoch_stats[1][j] += weight
        new_weight = retained + step_size * weight
        centers[j] = (retained * centers[j] + step_size * sums) / new_weight
        weight_ema[j] = new_weight
    return inertia


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("accumulate", [False, True])
def test_emeans_step_matches_reference(csr_container, dtype, accumulate):
    rng = np.random.RandomState(0)
    n_samples, n_features, n_clusters, batch_size = 1000, 7, 13, 600
    X = rng.normal(size=(n_samples, n_features)).astype(dtype)
    if csr_container is not None:
        X[X < 0.5] = 0
        X = csr_container(X)
    sample_weight = rng.uniform(size=n_samples).astype(dtype)
    sample_weight[:100] = 0
    centers = rng.normal(size=(n_clusters, n_features)).astype(dtype)
    weight_ema = rng.uniform(size=n_clusters)
    batch_indices = rng.permutation(n_samples)[:batch_size]

    def epoch_stats():
        if not accumulate:
            return None
        return [np.ones((n_clusters, n_features)), np.ones(n_clusters)]

    centers_ref, weight_ema_ref, stats_ref = (
        centers.copy(),
        weight_ema.copy(),
        epoch_stats(),
    )
    inertia_ref = _reference_step(
        X, sample_weight, batch_indices, centers_ref, weight_ema_ref, 0.3, stats_ref
    )

    step = _emeans_step_dense if csr_container is None else _emeans_step_sparse
    stats = epoch_stats() or [None, None]
    inertia = step(
        X,
        sample_weight,
        row_norms(X, squared=True),
        batch_indices,
        centers,
        weight_ema,
        0.3,
        *stats,
        2,
    )

    rtol = 1e-5 if dtype == np.float32 else 1e-10
    assert_allclose(inertia, inertia_ref, rtol=rtol)
    assert_allclose(centers, centers_ref, rtol=rtol, atol=rtol)
    assert_allclose(weight_ema, weight_ema_ref, rtol=rtol)
    if accumulate:
        assert_allclose(stats[0], stats_ref[0], rtol=rtol)
        assert_allclose(stats[1], stats_ref[1], rtol=rtol)


def test_emeans_step_full_step_size_gives_batch_means():
    # With a step size of 1, each center becomes the mean of its batch rows.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    centers = X[:4].copy()
    batch_indices = np.arange(200)
    labels = ((X[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
    _emeans_step_dense(
        X,
        np.ones(200),
        row_norms(X, squared=True),
        batch_indices,
        centers,
        np.full(4, 1e-5),
        1.0,
        None,
        None,
        1,
    )
    expected = np.array([X[labels == j].mean(axis=0) for j in range(4)])
    assert_allclose(centers, expected)


@pytest.mark.parametrize("csr_container", [None] + CSR_CONTAINERS)
def test_emeans_does_not_depend_on_n_threads(csr_container):
    X, _ = make_blobs(n_samples=5000, n_features=20, centers=30, random_state=0)
    if csr_container is not None:
        X[X < 0] = 0
        X = csr_container(X)
    emeans = EMeans(n_clusters=30, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        centers_sequential = emeans.fit(X).cluster_centers_
    centers_parallel = emeans.fit(X).cluster_centers_
    assert_array_equal(centers_sequential, centers_parallel)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_emeans_dense_sparse(csr_container):
    X, _ = make_blobs(n_samples=2000, n_features=10, centers=10, random_state=0)
    X[X < 0] = 0
    emeans_dense = EMeans(n_clusters=10, random_state=0).fit(X)
    emeans_sparse = EMeans(n_clusters=10, random_state=0).fit(csr_container(X))
    assert_allclose(emeans_dense.cluster_centers_, emeans_sparse.cluster_centers_)
    assert_array_equal(emeans_dense.labels_, emeans_sparse.labels_)


def test_emeans_float32_float64():
    X, _ = make_blobs(n_samples=2000, n_features=10, centers=10, random_state=0)
    emeans_64 = EMeans(n_clusters=10, random_state=0).fit(X)
    emeans_32 = EMeans(n_clusters=10, random_state=0).fit(X.astype(np.float32))
    assert emeans_32.cluster_centers_.dtype == np.float32
    assert_allclose(emeans_64.cluster_centers_, emeans_32.cluster_centers_, rtol=1e-4)


@pytest.mark.parametrize("n_features", [8, 32])
def test_emeans_inertia_close_to_kmeans(n_features):
    # Fewer epochs than KMeans iterations, for a similar inertia (within 2% in
    # practice).
    X, _ = make_blobs(
        n_samples=20_000,
        n_features=n_features,
        centers=200,
        cluster_std=3,
        random_state=0,
    )
    kmeans = KMeans(n_clusters=100, n_init=1, random_state=0).fit(X)
    emeans = EMeans(n_clusters=100, random_state=0).fit(X)
    assert emeans.n_iter_ < kmeans.n_iter_
    assert emeans.inertia_ < 1.03 * kmeans.inertia_


def test_emeans_recovers_blobs():
    X, y = make_blobs(n_samples=3000, n_features=5, centers=5, random_state=42)
    emeans = EMeans(n_clusters=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, emeans.labels_) == pytest.approx(1)


def test_emeans_labels_inertia_match_final_centers():
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=10, random_state=0)
    emeans = EMeans(n_clusters=10, random_state=0).fit(X)
    assert_array_equal(emeans.labels_, emeans.predict(X))
    assert emeans.inertia_ == pytest.approx(-emeans.score(X))


def test_emeans_revives_dead_clusters():
    # All initial centers are the same point: only the first one gets
    # samples, and the others are revived by splitting it.
    X, _ = make_blobs(n_samples=5000, n_features=5, centers=10, random_state=0)
    init = np.repeat(X[:1], 10, axis=0)
    emeans = EMeans(n_clusters=10, init=init, random_state=0).fit(X)
    assert len(np.unique(emeans.cluster_centers_, axis=0)) == 10
    assert np.all(np.bincount(emeans.labels_, minlength=10) > 0)


def test_revive_dead_clusters():
    rng = np.random.RandomState(0)
    centers = rng.uniform(1, 2, size=(4, 3))
    original_centers = centers.copy()
    original_weight_ema = np.array([10.0, 0.01, 20.0, 30.0])
    weight_ema = original_weight_ema.copy()

    assert _revive_dead_clusters(centers, weight_ema, rng) == 1

    # The dead cluster 1 takes half of a donor, which keeps the other half.
    (donor,) = [j for j in (0, 2, 3) if weight_ema[j] != original_weight_ema[j]]
    assert weight_ema[1] == weight_ema[donor] == original_weight_ema[donor] / 2
    assert not np.array_equal(centers[1], centers[donor])
    assert_allclose(centers[1], original_centers[donor], rtol=1e-3)
    assert_allclose((centers[1] + centers[donor]) / 2, original_centers[donor])
    others = [j for j in range(4) if j not in (1, donor)]
    assert_array_equal(centers[others], original_centers[others])


def test_revive_dead_clusters_null_center():
    centers = np.zeros((2, 3))
    weight_ema = np.array([1.0, 0.0])
    _revive_dead_clusters(centers, weight_ema, np.random.RandomState(0))
    assert not np.array_equal(centers[0], centers[1])
    assert_array_equal(weight_ema, [0.5, 0.5])


def test_scaling_laws():
    # Values of the reference implementation (faiss/contrib/e_means).
    assert _resolve_batch_size(1_000_000, 1024, 64, 5) == 65536
    assert _resolve_batch_size(1_000_000, 1024, 16, 5) == 8192
    assert _resolve_batch_size(100, 3, 2, 5) == 1
    assert _resolve_batch_size(10, 3, 64, 5) == 1
    assert _resolve_batch_size(12345, 100, 128, 2.5) == 512
    assert _resolve_initial_step_size(1_000_000, 1024, 5, 2048) == pytest.approx(
        0.8244 * 1024**-0.4617 * 2048**0.9003 * 1e6**-0.4498 * 5**-0.1057
    )
    assert _resolve_initial_step_size(1_000_000, 1024, 5, 65536) == 1


def test_emeans_batch_size():
    X = np.random.RandomState(0).normal(size=(100, 3))
    assert EMeans(n_clusters=3, batch_size=10).fit(X).batch_size_ == 10
    assert EMeans(n_clusters=3, batch_size=1000).fit(X).batch_size_ == 100
    emeans = EMeans(n_clusters=3, batch_size=30, n_epochs=3).fit(X)
    assert emeans.n_steps_ == 12
    assert emeans.n_iter_ == 3


@pytest.mark.parametrize(
    "n_epochs, expected",
    [(1, [100]), (3, [100, 100, 100]), (2.5, [100, 50, 100]), (0.25, [25])],
)
def test_epochs_n_rows(n_epochs, expected):
    assert _epochs_n_rows(100, n_epochs) == expected


def test_emeans_sample_weight_scale():
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=10, random_state=0)
    sample_weight = np.random.RandomState(0).uniform(size=1000)
    emeans = EMeans(n_clusters=10, random_state=0)
    centers = emeans.fit(X, sample_weight=sample_weight).cluster_centers_
    centers_scaled = emeans.fit(X, sample_weight=1e-6 * sample_weight).cluster_centers_
    assert_allclose(centers, centers_scaled)


def test_emeans_n_samples_lower_than_n_clusters():
    with pytest.raises(ValueError, match="n_samples=3 should be >= n_clusters=4"):
        EMeans(n_clusters=4).fit(np.zeros((3, 2)))
