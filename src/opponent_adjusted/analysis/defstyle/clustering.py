"""K-Means fitting, k-selection and stability diagnostics for CxG+ Phase B.

Preprocessing convention is inherited verbatim from Phase 2
(`scripts/materialize_cxg_defensive_profile_clusters.py`): a median
`SimpleImputer` followed by a `StandardScaler`, BOTH FITTED ON THE TRAIN
SPLIT ONLY and then applied unchanged to every other split. Validation and
test data never influence the imputation medians, the scaling statistics or
the centroids.

The split reused here is the project's existing match-level split
(`oam_analysis.cxg_match_splits_v1`, 426 train / 92 validation / 92 test
matches, produced by `scripts/run_cxg_split_analysis.py` and consumed by
`cxg_plus_360_model_matrix_v1`). No new split is invented. Because the
clustering unit is a PLAYER rather than a shot, "train" is applied at the
event level: a player's TRAIN vector is built only from events in
train-split matches.

Two vectors per player, deliberately:
  - the TRAIN vector (train-match events only) gates and drives the fit;
  - the DEPLOY vector (the player's full event history) is what gets scored
    through the train-fitted transform at assignment time.
The fit therefore never sees a validation/test-only player, while assignment
still uses the best available estimate of each player's style. This is the
player-level analogue of Phase 2's "fit on train rows, predict on all rows".

Note on imputation: the share vectors are constructed with a guaranteed
non-zero denominator (see `features.action_shares`), so no share is ever
NaN and the median imputer is a structural no-op here. It is retained
anyway so this phase's preprocessing is literally the same object graph as
Phase 2's, rather than a lookalike that would silently diverge if a future
feature did admit missingness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Matches run_cxg_split_analysis.py's split seed, the repo-wide reproducibility
# convention also followed by Phase 2.
SEED = 260821

K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

# Seeds used for the stability check. SEED itself is excluded (comparing the
# reference fit against itself would trivially score 1.0).
STABILITY_SEEDS: tuple[int, ...] = (1, 7, 42, 1234, 260822)

N_BOOTSTRAP = 25
BOOTSTRAP_FRACTION = 0.8


@dataclass(frozen=True)
class KDiagnostic:
    k: int
    silhouette_train: float
    inertia_train: float
    cluster_sizes_train: list[int]
    min_cluster_fraction_train: float


@dataclass(frozen=True)
class StabilityReport:
    """Adjusted Rand Index of re-fits against the reference fit's labels.

    ARI is used rather than raw label agreement because K-Means cluster
    indices are arbitrary: ARI is invariant to relabelling, so it measures
    whether the same PARTITION is recovered, not whether the same integers
    were handed out.
    """

    seed_ari: dict[int, float]
    bootstrap_ari_mean: float
    bootstrap_ari_min: float
    bootstrap_ari_values: list[float] = field(default_factory=list)
    holdout_split_ari: dict[str, float] = field(default_factory=dict)

    @property
    def min_seed_ari(self) -> float:
        return min(self.seed_ari.values()) if self.seed_ari else float("nan")


def build_preprocessor() -> Pipeline:
    """Median imputation + standardisation, exactly as Phase 2."""
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )


def k_diagnostics(x_train: np.ndarray, k_range: tuple[int, ...] = K_RANGE) -> list[KDiagnostic]:
    """Silhouette + elbow(inertia) + size-balance per k, on the train matrix."""
    diagnostics: list[KDiagnostic] = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(x_train)
        sizes = np.bincount(model.labels_, minlength=k)
        diagnostics.append(
            KDiagnostic(
                k=k,
                silhouette_train=float(silhouette_score(x_train, model.labels_)),
                inertia_train=float(model.inertia_),
                cluster_sizes_train=sizes.tolist(),
                min_cluster_fraction_train=float(sizes.min() / sizes.sum()),
            )
        )
    return diagnostics


def fit_kmeans(x_train: np.ndarray, k: int, seed: int = SEED) -> KMeans:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x_train)


def stability_report(
    x_train: np.ndarray,
    k: int,
    reference: KMeans | None = None,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_fraction: float = BOOTSTRAP_FRACTION,
) -> StabilityReport:
    """Seed-perturbation and subsample-perturbation stability of the partition.

    Two independent perturbations, because they fail for different reasons:
      - re-seeding tests whether the solution is an artefact of one lucky
        K-Means initialisation;
      - re-fitting on random 80% subsamples tests whether the solution is an
        artefact of the particular players in the train split. Bootstrap ARI
        is measured on the intersection (the subsampled rows), since the
        refit only has an opinion about the rows it saw.
    """
    model = reference if reference is not None else fit_kmeans(x_train, k)
    base_labels = model.predict(x_train)

    seed_ari = {
        seed: float(adjusted_rand_score(base_labels, fit_kmeans(x_train, k, seed).predict(x_train)))
        for seed in seeds
    }

    rng = np.random.default_rng(SEED)
    n_rows = x_train.shape[0]
    size = max(k, int(round(n_rows * bootstrap_fraction)))
    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_rows, size=size, replace=False)
        refit = fit_kmeans(x_train[idx], k)
        bootstrap_values.append(
            float(adjusted_rand_score(base_labels[idx], refit.predict(x_train[idx])))
        )

    return StabilityReport(
        seed_ari=seed_ari,
        bootstrap_ari_mean=float(np.mean(bootstrap_values)) if bootstrap_values else float("nan"),
        bootstrap_ari_min=float(np.min(bootstrap_values)) if bootstrap_values else float("nan"),
        bootstrap_ari_values=bootstrap_values,
    )


def holdout_refit_ari(
    x_train: np.ndarray, x_holdout: np.ndarray, x_all: np.ndarray, k: int
) -> float:
    """ARI between the train-fit and an independent holdout-fit, scored on all rows.

    This is the strongest of the three checks: it refits the model from
    scratch on data the reference model never touched, then asks whether the
    two models carve the SAME population the same way. A high score means
    the archetypes are a property of football, not of the train split.
    """
    train_model = fit_kmeans(x_train, k)
    holdout_model = fit_kmeans(x_holdout, k)
    return float(adjusted_rand_score(train_model.predict(x_all), holdout_model.predict(x_all)))
