# V1 Release Scope

## Purpose

V1 should present the repository as a reviewable football analytics product foundation. It should make the implemented modelling paths reproducible, explain their outputs clearly, and define the dashboard/storytelling layer needed for portfolio review.

V1 is not a production deployment and not the end of the modelling roadmap.

## Included In V1

- CxG implemented with feature generation, baseline modelling, validation reporting, output contract checks, model metadata, and API-backed inference.
- CxA implemented with feature generation, baseline modelling, attribution, player/team/sequence aggregates, and generated reports.
- Baseline CxT implemented with deterministic zone/grid threat values.
- CxT aggregate and interpretation reports implemented, including player/team/sequence aggregates, zone-transition summaries, top-action reports, and interpretation summary.
- Generated outputs under `feature_store/` and `outputs/` remain ignored by Git.
- Feature contracts and dashboard data contracts define expected generated inputs.
- Tests and CI quality gates cover contracts, reproducibility checks, API integration, baseline outputs, and generated-output ignore rules.
- Streamlit dashboard v1 shell implemented for portfolio/demo review.
- Dashboard and storytelling docs define the v1 product surface and reviewer walkthrough.

## Excluded From V1

- CxT+ is not implemented in v1.
- Contextual CxT is not implemented in v1.
- Advanced CxT is not implemented in v1.
- OD-CxT is not implemented in v1.
- OD-CxT+ is not implemented in v1.
- Production deployment is not included in v1.
- Live data ingestion is not included in v1.
- Tracking data is not required or included in v1.
- Advanced dashboard visual design is not included in v1.
- Production-grade calibration, monitoring, and model registry workflows are not claimed.

## V1 Dashboard Surface

The v1 dashboard makes the existing outputs understandable quickly. It focuses on:

- project overview
- player analysis
- team analysis
- CxG analysis
- CxA analysis
- baseline CxT analysis
- action-level explorer
- model/report diagnostics
- example insights

The dashboard should read from generated outputs rather than committing generated data to the repository.

## Release Readiness Checklist

- CxG outputs can be regenerated locally.
- CxA outputs can be regenerated locally.
- Baseline CxT outputs can be regenerated locally.
- Generated outputs are ignored by Git.
- Dashboard contract references the expected generated paths.
- README describes the v1 product direction honestly.
- Validation commands pass locally and in CI.

## Post-V1 Roadmap

Post-v1 work can revisit more ambitious modelling and product layers:

- CxT+ with richer action context.
- Contextual CxT.
- Advanced CxT state-value modelling.
- OD-CxT and OD-CxT+.
- Full dashboard build.
- More robust model cards and calibration narratives.
- Production-style serving, monitoring, and deployment workflows.
