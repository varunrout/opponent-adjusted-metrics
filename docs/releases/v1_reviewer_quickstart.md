# V1 Reviewer Quickstart

This path is designed for a 5-10 minute review of the project.

## 1. Clone And Install

```bash
git clone https://github.com/varunrout/opponent-adjusted-metrics.git
cd opponent-adjusted-metrics
poetry install
```

## 2. Run The Test Suite

```bash
poetry run pytest -v -m "not e2e"
```

For the full local quality gate, also run:

```bash
poetry run ruff check src scripts tests app
poetry run black --check src scripts tests app
```

## 3. Launch The Dashboard

```bash
make dashboard
```

Equivalent direct command:

```bash
poetry run streamlit run app/streamlit_app.py
```

## 4. Understand Missing-Output Behaviour

The dashboard reads generated files from `feature_store/` and `outputs/`. Those files are intentionally ignored by Git. In a clean checkout, some dashboard sections may show empty tables and missing-output guidance. That is expected.

To populate the dashboard locally, run:

```bash
make cxg-smoke
make cxa-smoke
make cxt-baseline
```

## 5. Review The Project Story

Recommended docs:

- [v1 release notes](v1.0.0.md)
- [v1 release scope](v1_scope.md)
- [dashboard demo walkthrough](../dashboard/demo_walkthrough.md)
- [project story](../storytelling/v1_project_story.md)
- [generated outputs](../OUTPUTS.md)

## 6. What To Look For

- CxG explains shot quality.
- CxA explains chance-creation action value.
- Baseline CxT explains threat added by ball progression.
- Player/team/action pages show how modelling outputs become interpretable football insight.
- Reports and diagnostics show which generated outputs exist locally.

## 7. What Is Deferred

CxT+, Contextual CxT, Advanced CxT, OD-CxT, production deployment, live data ingestion, and tracking-data workflows are outside v1.
