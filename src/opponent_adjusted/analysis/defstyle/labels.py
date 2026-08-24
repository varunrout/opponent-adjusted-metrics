"""Deriving interpretable archetype labels from K-Means centroids (CxG+ Phase B).

Labels are DERIVED from the fitted centroids, never hand-assigned to a
cluster index. The rule is deliberately mechanical so that a cluster which
does not actually look like an archetype cannot be dressed up as one:

  1. Read the centroid in SCALED space (the K-Means centres themselves),
     i.e. how many cohort standard deviations each action share sits from
     the cohort mean. Scaled space rather than raw share space is essential
     here: raw shares are dominated by Pressure (~60% of all qualifying
     actions), so a raw-share argmax would call every cluster a "presser",
     while a raw LIFT ratio would over-reward the rarest category. The
     z-score asks the right question -- which action is unusually
     over-represented relative to how much that action normally varies.
  2. The archetype is named after the single most over-represented action.
  3. If even that action is under `MIN_DISTINCTIVENESS_Z` standard
     deviations above the mean, the cluster has no distinguishing behaviour
     and is labelled muddy rather than given a flattering name.
  4. If the most over-represented action is 50/50, the cluster is labelled
     muddy for a specific, documented reason (see below) rather than being
     named "50/50 specialist", which would imply a playing style the data
     does not support.

THE 50/50 CAVEAT
----------------
50/50 is 0.4% of all qualifying actions, and its recording rate varies ~6x
across the competitions in this corpus (measured live: 0.21% of qualifying
events in competition 55/season 43 up to 1.29% in competition 55/season
282). A cluster whose defining characteristic is an elevated 50/50 share is
therefore substantially tracking WHICH COMPETITION a player appears in, not
how they defend -- and indeed the k=4 solution's 50/50-led cluster draws
91% of its members from competitions 43 and 55 while being positionally
mixed (48% midfielders, 27% forwards, 25% defenders), unlike the three
clean clusters which are each positionally coherent. It is reported as
muddy, honestly, rather than narrated into an archetype.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from opponent_adjusted.analysis.defstyle.features import STYLE_FEATURES

MIN_DISTINCTIVENESS_Z = 0.5

MUDDY_LABEL = "unresolved_mixed_profile"
MUDDY_5050_LABEL = "unresolved_5050_annotation_density"

# Most over-represented action share -> archetype label.
ARCHETYPE_BY_DOMINANT_FEATURE: dict[str, str] = {
    "pressure_share": "high_volume_presser",
    "duel_share": "duel_dominant_contester",
    "interception_share": "anticipatory_interceptor",
    "clearance_share": "deep_block_clearer",
    "block_share": "shot_blocker",
    "foul_committed_share": "disruptive_fouler",
    # `fifty_fifty_share` is intentionally absent: it routes to MUDDY_5050_LABEL.
}


def derive_archetype_label(scaled_centroid: Mapping[str, float] | Sequence[float]) -> str:
    """Name one cluster from its centroid in scaled (z-score) space.

    Accepts either a mapping keyed by `STYLE_FEATURES` or a bare sequence in
    `STYLE_FEATURES` order.
    """
    if isinstance(scaled_centroid, Mapping):
        values = {feature: float(scaled_centroid[feature]) for feature in STYLE_FEATURES}
    else:
        if len(scaled_centroid) != len(STYLE_FEATURES):
            raise ValueError(
                f"expected {len(STYLE_FEATURES)} centroid components, got {len(scaled_centroid)}"
            )
        values = dict(zip(STYLE_FEATURES, (float(v) for v in scaled_centroid), strict=True))

    dominant = max(values, key=lambda feature: values[feature])
    if values[dominant] < MIN_DISTINCTIVENESS_Z:
        return MUDDY_LABEL
    if dominant == "fifty_fifty_share":
        return MUDDY_5050_LABEL
    return ARCHETYPE_BY_DOMINANT_FEATURE[dominant]


def derive_archetype_labels(scaled_centroids: Sequence[Sequence[float]]) -> list[str]:
    """Name every cluster, disambiguating any repeated label by rank.

    Two clusters can legitimately be led by the same action (e.g. a moderate
    and an extreme presser). Rather than collapsing them to one name, the
    more extreme one keeps the base label and the others are suffixed
    `_secondary`, `_tertiary`, ... in descending order of that action's
    z-score, so every label in the profile table stays unique.
    """
    labels = [derive_archetype_label(centroid) for centroid in scaled_centroids]
    suffixes = ("", "_secondary", "_tertiary", "_quaternary")

    out = list(labels)
    for label in set(labels):
        members = [i for i, value in enumerate(labels) if value == label]
        if len(members) == 1:
            continue
        peak = {i: max(scaled_centroids[i]) for i in members}
        for rank, index in enumerate(sorted(members, key=lambda i: -peak[i])):
            suffix = suffixes[rank] if rank < len(suffixes) else f"_{rank + 1}"
            out[index] = f"{label}{suffix}"
    return out


def is_muddy(label: str) -> bool:
    """True for labels that explicitly decline to claim an interpretation."""
    return label.startswith(MUDDY_LABEL) or label.startswith(MUDDY_5050_LABEL)
