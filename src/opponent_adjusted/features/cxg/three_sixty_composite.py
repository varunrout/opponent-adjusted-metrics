"""CxG+ F15: composite spatial-state features (cxg_360_context_v1).

Implemented last, after the F1-F14 components they read are already governed.
Every composite below uses fixed, documented, non-target-tuned arithmetic
composition of already-computed F-values -- no weights are fit against goal
outcome or any other target.
"""

from __future__ import annotations

from opponent_adjusted.features.cxg.contracts import three_sixty_candidate_names_for_families

F15_FEATURES = three_sixty_candidate_names_for_families(("F15",))

# Fixed, versioned parameters (cxg_360_context_v1). COMPACTNESS_NORMALIZER is a coarse
# quarter-pitch-area scale (120 * 80 / 4 native units^2), not fit against any target.
COMPACTNESS_NORMALIZER = 2400.0


def derive_composite_360_context(
    dynamic_values: dict[str, object | None],
    static_values: dict[str, object | None],
    possession_age_s: float | None,
) -> dict[str, object | None]:
    values: dict[str, object | None] = {name: None for name in F15_FEATURES}

    nearest_delta = dynamic_values.get("nearest_defender_distance_delta")
    if nearest_delta is not None and possession_age_s is not None and possession_age_s > 0:
        values["transition_space_decay"] = -nearest_delta / possession_age_s

    reset_fraction = dynamic_values.get("rest_defence_reset_fraction")
    if reset_fraction is not None:
        compactness_delta = dynamic_values.get("defensive_compactness_delta")
        penalty = (
            max(0.0, compactness_delta) / COMPACTNESS_NORMALIZER
            if compactness_delta is not None
            else 0.0
        )
        values["defensive_reset_index"] = reset_fraction - penalty

    total_displacement = dynamic_values.get("gk_total_displacement")
    if total_displacement is not None:
        values["gk_setness_proxy"] = 1.0 / (1.0 + total_displacement)

    return values
