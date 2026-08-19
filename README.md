# Opponent-Adjusted Metrics Productionisation

Production GCP foundation for opponent-adjusted football analytics using StatsBomb Open Data.

## Current Implemented Scope

- Pinned StatsBomb Open Data ingestion
- Selected competition subset
- StatsBomb 360 coverage ingestion and audit
- Immutable Bronze landing in GCS
- Typed Silver transformation
- `oam_core` BigQuery publication
- Terraform production data foundation

## Repository Structure

- `configs/statsbomb_subset.json`
- `infra/terraform/`
- `scripts/`
- `src/opponent_adjusted/ingestion/`
- `src/opponent_adjusted/storage/`
- `src/opponent_adjusted/pipelines/silver/`
- `tests/`

CxG, CxG+, CxA, CxT, and other model-family implementation are not present in the current production tree yet. Historical implementation is preserved in Git history and the `pre-gcp-productionisation-2026-08-19` tag.

Detailed architecture and control-plane documentation lives outside this repository and is not duplicated here.

## Validation

```bash
poetry install
poetry run pytest -q
```
