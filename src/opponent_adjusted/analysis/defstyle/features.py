"""Feature construction for defender playing-STYLE clustering (CxG+ Phase B).

Source table: `oam_core.events` -- a player's ENTIRE event history, not the
shot-facing "nearest defender" involvement rows used by the deprecated ODI
feature. That distinction is the whole point of this phase:

  - The deprecated `nearest_defender_odi` was built from
    `oam_analysis.cxg_defensive_involvement_v1` (one row per shot faced),
    giving a median of ~4 observations per player -- far too sparse to
    characterise anything, and it consumed `statsbomb_xg` as an input, which
    is circular with this project's own xG benchmarking. Neither
    `cxg_defensive_involvement_v1.statsbomb_xg` nor `shots.statsbomb_xg` is
    read anywhere in this module or its pipeline.
  - This phase instead reads every defensive-action event the player has
    ever produced. Measured live on 2026-08-22 against
    `oam_core.events` (`silver_schema_version='statsbomb_silver_v1_2'`,
    `data_version='b0bc9f22...'`, joined to the 610 split-assigned matches):
    1,958 players, 316,808 qualifying events, median 66 / mean 162 per
    player (p25=22, p75=167, max=2,078). Note this measured median of 66 is
    materially lower than the "~186" figure quoted in the task brief; the
    brief's conclusion (sample size is not the binding constraint here)
    still holds -- 66 is an order of magnitude better than ODI's ~4 -- but
    the number itself is recorded honestly rather than restated.

ACTION TAXONOMY
---------------
Seven `event_type_name` values, all of them defensive actions attributable
to an individual player. Duel is a SINGLE combined rate, not split by
sub-type, because duel sub-type is genuinely unavailable in this warehouse:

  - `oam_core.events` carries no duel outcome/sub-type column (verified
    against the live schema: the only outcome-ish flat columns are
    `under_pressure`, `counterpress`, `off_camera`, `out`).
  - There is no qualifiers side-table and no STRUCT/JSON column anywhere in
    `oam_core` (18 tables; `duels` is not among them, and the Silver
    contract in `pipelines/silver/contracts.py` defines no duel table).
  - Sub-typed detail tables DO exist for other families (`passes`,
    `dribbles`, `pressures`, `disciplinary_events` all carry
    `outcome_id`/`outcome_name`), so the absence for duels is a real gap in
    the Silver projection, not an oversight in this search.

Aerial-vs-ground and won-vs-lost duel splits are therefore out of reach
without re-ingesting raw StatsBomb JSON, which is out of scope here.

NORMALISATION -- shares, not per-90
-----------------------------------
Each player's vector is the COMPOSITIONAL SHARE of their qualifying
defensive actions falling into each of the seven types (the seven shares
sum to 1). Per-90 was considered and deliberately rejected:

  - The question is a player's style, i.e. the MIX of defensive actions
    they choose, not how much defending they do. Per-90 rates conflate the
    two: a 90-minute centre-back and a 20-minute substitute centre-back with
    identical habits get different per-90 vectors purely from volume and
    from their team's share of possession.
  - Per-90 additionally requires per-player minutes reconstructed from
    `starting_xi_players` + `substitutions`, which introduces its own
    estimation error (and the roster resolver's documented period-boundary
    convention, see `analysis/odi/roster.py`) for no gain on the style
    question.
  - Shares are already scale-free, so the volume signal is carried purely by
    the sample-size threshold below rather than leaking into the features.

The seven shares are linearly dependent (they sum to 1), so the vectors lie
on a 6-simplex. That is harmless for K-Means -- it is a rank deficiency in
the ambient space, not a degeneracy of Euclidean distance on the simplex --
and it is preferable to dropping one category arbitrarily, which would make
the geometry depend on which category was chosen as the reference.

SAMPLE-SIZE THRESHOLD
---------------------
`MIN_ACTIONS = 30` qualifying events, chosen from the observed distribution
rather than assumed:

  - The task brief proposed >=20 as a starting point. At n=20 a share near
    the cohort mean for a mid-frequency category (e.g. duel, ~0.14) has a
    binomial standard error of ~0.078 -- more than half the mean itself.
    At n=30 that falls to ~0.063, and the dominant category's share (Pressure,
    ~0.60) has SE <= 0.09.
  - The measured cost/benefit turns at 30. Raising 20 -> 30 costs only 2.5
    percentage points of CxG+ shot coverage (92.3% -> 89.8%); raising
    30 -> 50 costs a further 7.1 points (89.8% -> 82.7%) for a much smaller
    noise reduction. 30 is the knee.
  - 30 sits at roughly the 35th percentile of the per-player count
    distribution (median 66), so it excludes the genuinely unobservable tail
    without discarding the ordinary squad player.

Players below the threshold are NOT clustered and NOT given a fallback or
default archetype; they surface as NULL with an explicit reason.
"""

from __future__ import annotations

from collections.abc import Mapping

# StatsBomb `event_type_name` -> feature-column name. Order is the canonical
# feature-vector order used everywhere downstream (centroids, charts, tests).
ACTION_TYPE_TO_FEATURE: dict[str, str] = {
    "Pressure": "pressure_share",
    "Duel": "duel_share",
    "Interception": "interception_share",
    "Clearance": "clearance_share",
    "Block": "block_share",
    "Foul Committed": "foul_committed_share",
    "50/50": "fifty_fifty_share",
}

ACTION_TYPES: tuple[str, ...] = tuple(ACTION_TYPE_TO_FEATURE)
STYLE_FEATURES: tuple[str, ...] = tuple(ACTION_TYPE_TO_FEATURE.values())

FEATURE_SET_VERSION = "cxg_defstyle_action_mix_v1_s7"

MIN_ACTIONS = 30

# Explicit NULL reasons. Never a default/fallback archetype.
NULL_REASON_BELOW_THRESHOLD = "below_min_action_threshold"
NULL_REASON_NO_DEFENDER = "no_freeze_frame_defender_resolved"
NULL_REASON_NOT_CLUSTERED = "defender_not_in_cluster_table"

# Duel sub-type availability, established live -- see module docstring.
DUEL_SUBTYPE_AVAILABLE = False


def meets_threshold(total_actions: int) -> bool:
    """True when a player has enough qualifying actions to be clustered."""
    return total_actions >= MIN_ACTIONS


def total_actions(counts: Mapping[str, int]) -> int:
    """Total qualifying defensive actions across the seven tracked types.

    Keys not in `ACTION_TYPES` are ignored rather than silently summed, so a
    caller passing a wider count dict cannot inflate the denominator.
    """
    return sum(int(counts.get(action, 0) or 0) for action in ACTION_TYPES)


def action_shares(counts: Mapping[str, int]) -> dict[str, float] | None:
    """Compositional action-mix vector for one player, or None if too sparse.

    `counts` is keyed by StatsBomb `event_type_name`. Returns a dict keyed by
    `STYLE_FEATURES` whose values sum to 1.0, or None when the player falls
    below `MIN_ACTIONS` -- None means "do not cluster this player", and the
    caller is responsible for recording `NULL_REASON_BELOW_THRESHOLD`.
    """
    total = total_actions(counts)
    if not meets_threshold(total):
        return None
    return {
        feature: int(counts.get(action, 0) or 0) / total
        for action, feature in ACTION_TYPE_TO_FEATURE.items()
    }


def feature_vector(shares: Mapping[str, float]) -> list[float]:
    """Order a share mapping into the canonical `STYLE_FEATURES` sequence."""
    return [float(shares[feature]) for feature in STYLE_FEATURES]
