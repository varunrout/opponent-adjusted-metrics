# Ingestion Reproducibility

This document describes the reproducible data ingestion path for `opponent-adjusted-metrics`.

## Goal

A fresh clone should be able to fetch the configured StatsBomb Open Data subset, load it into the local database, normalise events, and produce a row-count report.

## Configured data subset

The default subset is stored in:

```text
configs/statsbomb_subset.json
```

It currently includes:

- Premier League 2015/16
- FIFA World Cup 2018
- FIFA World Cup 2022
- UEFA Euro 2020
- UEFA Euro 2024

The subset file records competition IDs, season IDs, labels, event inclusion, and purpose. This makes the project less dependent on implicit hardcoded filters.

## Fresh database path

```bash
cp .env.example .env
make db-up
make migrate-up
make fetch-data
make ingest-all
make normalize-events
make ingestion-report
```

## One-command data smoke path

After database setup and migrations:

```bash
make data-smoke
```

This runs:

1. fetch configured StatsBomb files
2. ingest competitions
3. ingest matches
4. ingest raw events
5. normalise events
6. write ingestion status report

## Fetching data

Default command:

```bash
make fetch-data
```

Use a custom config:

```bash
make fetch-data CONFIG=configs/statsbomb_subset.json
```

Force re-download:

```bash
make fetch-data FORCE=1
```

The fetcher writes a summary to:

```text
outputs/reports/ingestion/fetch_summary.json
```

## Reporting database status

Run:

```bash
make ingestion-report
```

The report is written to:

```text
outputs/reports/ingestion/db_status.json
```

It includes:

- table counts
- top raw event types
- readiness flags for competitions, matches, raw events, normalised events, shots, features, registry, and predictions

## Idempotency expectations

The ingestion scripts are expected to be rerunnable. Existing records should be skipped rather than duplicated. The report script should be used after reruns to confirm row counts remain stable where no new source data was added.

## Remaining hardening work

Fixture-backed ingestion tests exist. Follow-up data work should add:

- duplicate detection reports
- normalisation coverage by event type
- failed-match retry logs
- configurable small smoke subset for CI
- checks comparing expected match files with loaded match rows
- full fresh-clone validation against Docker Postgres
