import numpy as np
import pytest

from opponent_adjusted.analysis.defstyle.clustering import (
    build_preprocessor,
    fit_kmeans,
    holdout_refit_ari,
    k_diagnostics,
    stability_report,
)
from opponent_adjusted.analysis.defstyle.features import STYLE_FEATURES
from opponent_adjusted.analysis.defstyle.labels import (
    MIN_DISTINCTIVENESS_Z,
    MUDDY_5050_LABEL,
    MUDDY_LABEL,
    derive_archetype_label,
    derive_archetype_labels,
    is_muddy,
)


def _synthetic_three_group_matrix(seed: int = 0) -> np.ndarray:
    """Three well-separated action-mix archetypes, 60 players each."""
    rng = np.random.default_rng(seed)
    presser = [0.80, 0.06, 0.03, 0.03, 0.03, 0.04, 0.01]
    clearer = [0.35, 0.10, 0.08, 0.30, 0.10, 0.06, 0.01]
    dueller = [0.45, 0.35, 0.03, 0.05, 0.04, 0.07, 0.01]
    rows = []
    for centre in (presser, clearer, dueller):
        block = np.array(centre) + rng.normal(0, 0.012, size=(60, len(STYLE_FEATURES)))
        rows.append(np.clip(block, 1e-6, None))
    matrix = np.vstack(rows)
    return matrix / matrix.sum(axis=1, keepdims=True)


def test_preprocessor_fits_on_train_only_and_transforms_holdout_unchanged():
    train = _synthetic_three_group_matrix(seed=1)
    holdout = _synthetic_three_group_matrix(seed=2)
    pre = build_preprocessor().fit(train)

    scaler = pre.named_steps["scale"]
    before = (scaler.mean_.copy(), scaler.scale_.copy())
    pre.transform(holdout)
    after = (scaler.mean_, scaler.scale_)

    # Transforming holdout data must never refit the scaling statistics.
    assert np.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])
    # Train is standardised by construction; holdout is not forced to be.
    assert np.allclose(pre.transform(train).mean(axis=0), 0.0, atol=1e-9)


def test_median_imputer_fills_from_train_medians_only():
    train = _synthetic_three_group_matrix(seed=1)
    pre = build_preprocessor().fit(train)
    expected_median = np.median(train[:, 0])

    holdout = train[:2].copy()
    holdout[0, 0] = np.nan
    imputed = pre.named_steps["impute"].transform(holdout)
    assert imputed[0, 0] == pytest.approx(expected_median)


def test_k_diagnostics_reports_every_k_with_balanced_size_accounting():
    train = build_preprocessor().fit_transform(_synthetic_three_group_matrix())
    diagnostics = k_diagnostics(train, (2, 3, 4))
    assert [d.k for d in diagnostics] == [2, 3, 4]
    for d in diagnostics:
        assert len(d.cluster_sizes_train) == d.k
        assert sum(d.cluster_sizes_train) == train.shape[0]
        assert 0.0 < d.min_cluster_fraction_train <= 1.0
    # Inertia must fall monotonically as k grows.
    assert diagnostics[0].inertia_train > diagnostics[1].inertia_train > diagnostics[2].inertia_train


def test_k_diagnostics_recovers_the_planted_group_count():
    train = build_preprocessor().fit_transform(_synthetic_three_group_matrix())
    diagnostics = {d.k: d.silhouette_train for d in k_diagnostics(train, (2, 3, 4))}
    assert diagnostics[3] == max(diagnostics.values())


def test_stability_is_high_on_well_separated_data():
    train = build_preprocessor().fit_transform(_synthetic_three_group_matrix())
    report = stability_report(train, k=3, n_bootstrap=5)
    assert report.min_seed_ari > 0.95
    assert report.bootstrap_ari_mean > 0.95
    assert len(report.bootstrap_ari_values) == 5


def test_stability_collapses_on_structureless_data():
    # Isotropic noise has no clusters; a refit must NOT reproduce the partition.
    # This guards against the check being vacuously satisfiable.
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(180, len(STYLE_FEATURES)))
    report = stability_report(noise, k=3, n_bootstrap=5)
    assert report.bootstrap_ari_mean < 0.8


def test_holdout_refit_ari_agrees_when_both_splits_share_structure():
    train = _synthetic_three_group_matrix(seed=1)
    holdout = _synthetic_three_group_matrix(seed=2)
    everything = np.vstack([train, holdout])
    pre = build_preprocessor().fit(train)
    ari = holdout_refit_ari(
        pre.transform(train), pre.transform(holdout), pre.transform(everything), k=3
    )
    assert ari > 0.95


def test_fit_is_deterministic_for_a_fixed_seed():
    train = build_preprocessor().fit_transform(_synthetic_three_group_matrix())
    first = fit_kmeans(train, 3)
    second = fit_kmeans(train, 3)
    assert np.allclose(first.cluster_centers_, second.cluster_centers_)


def test_label_is_named_after_the_most_over_represented_action():
    centroid = dict.fromkeys(STYLE_FEATURES, -0.2)
    centroid["clearance_share"] = 1.4
    assert derive_archetype_label(centroid) == "deep_block_clearer"


def test_label_uses_z_space_not_raw_share_dominance():
    # Pressure is the largest RAW share in almost every real centroid; a
    # clearance-led archetype must not be misnamed a presser.
    centroid = dict.fromkeys(STYLE_FEATURES, 0.0)
    centroid["pressure_share"] = -1.18
    centroid["clearance_share"] = 1.22
    assert derive_archetype_label(centroid) == "deep_block_clearer"


def test_indistinct_centroid_is_labelled_muddy_not_named():
    centroid = dict.fromkeys(STYLE_FEATURES, MIN_DISTINCTIVENESS_Z - 0.01)
    assert derive_archetype_label(centroid) == MUDDY_LABEL
    assert is_muddy(derive_archetype_label(centroid))


def test_fifty_fifty_led_cluster_is_muddy_by_annotation_density():
    centroid = dict.fromkeys(STYLE_FEATURES, -0.1)
    centroid["fifty_fifty_share"] = 2.23
    label = derive_archetype_label(centroid)
    assert label == MUDDY_5050_LABEL
    assert is_muddy(label)


def test_label_accepts_a_bare_sequence_in_canonical_order():
    values = [0.0] * len(STYLE_FEATURES)
    values[STYLE_FEATURES.index("duel_share")] = 1.5
    assert derive_archetype_label(values) == "duel_dominant_contester"


def test_label_rejects_a_wrong_length_sequence():
    with pytest.raises(ValueError):
        derive_archetype_label([0.0, 1.0])


def test_repeated_dominant_action_yields_unique_ranked_labels():
    strong = [0.0] * len(STYLE_FEATURES)
    strong[STYLE_FEATURES.index("pressure_share")] = 2.0
    weak = [0.0] * len(STYLE_FEATURES)
    weak[STYLE_FEATURES.index("pressure_share")] = 0.9
    labels = derive_archetype_labels([weak, strong])
    assert len(set(labels)) == 2
    assert labels[1] == "high_volume_presser"
    assert labels[0] == "high_volume_presser_secondary"


def test_production_centroids_reproduce_the_reported_archetypes():
    # Scaled centroids of the shipped k=4 model (see
    # audit_outputs/.../raw/cluster_profile.json), in STYLE_FEATURES order.
    centroids = [
        [0.30, -0.28, -0.09, -0.43, -0.05, 0.03, 2.23],
        [-0.18, 1.51, -0.60, -0.36, -0.46, 0.33, -0.35],
        [0.73, -0.51, -0.27, -0.54, -0.26, 0.11, -0.26],
        [-1.18, 0.12, 0.77, 1.22, 0.67, -0.36, -0.32],
    ]
    assert derive_archetype_labels(centroids) == [
        MUDDY_5050_LABEL,
        "duel_dominant_contester",
        "high_volume_presser",
        "deep_block_clearer",
    ]
