# Dashboard Demo Walkthrough

## Goal

Use this walkthrough to demo the v1 Streamlit dashboard to a reviewer. The point is to show a clear football analytics product story, not to claim production deployment or advanced CxT modelling.

## Before The Demo

Install dependencies and regenerate any outputs you want to show:

```bash
poetry install
make cxg-smoke
make cxa-smoke
make cxt-baseline
```

Then launch the dashboard:

```bash
make dashboard
```

Direct command:

```bash
poetry run streamlit run app/streamlit_app.py
```

If generated outputs are missing, the dashboard still opens and shows availability guidance. That is expected in a clean checkout.

## Suggested Reviewer Walkthrough

1. Open **Overview**.
   Explain what problem the project solves: raw event data is converted into contextual football metrics and interpretable reports.

2. Read the **V1 Status** section.
   Point out what is implemented: CxG, CxA, baseline CxT, dashboard shell, and aggregate/report views.

3. Point out deferred work.
   CxT+, Contextual CxT, Advanced CxT, and OD-CxT / OD-CxT+ are intentionally deferred until after v1.

4. Open **Player analysis**.
   Explain that high CxG means shot quality, high CxA means chance-creation action value, and high baseline CxT means repeated progression into more dangerous areas.

5. Open **Team analysis**.
   Compare teams by how they create value: shots, chance creation, or territorial progression.

6. Open **CxG**.
   Say: "CxG evaluates shot quality, not whether the shot became a goal."

7. Open **CxA**.
   Say: "A high CxA player contributes actions that move possessions closer to chance creation."

8. Open **CxT**.
   Say: "Baseline CxT is location-threat movement, not full possession-state value."

9. Open **Action explorer**.
   Show that aggregate values can be traced back to individual actions.

10. Open **Reports / diagnostics**.
    Show generated-output availability and report metadata, including missing-output behavior.

## Screenshot / GIF Targets

Future release screenshots or GIFs should show:

- Overview page with v1 status and metric explanations.
- Player analysis with top CxT or CxA leaderboards.
- Team analysis with aggregate comparison.
- CxT page with sequence, zone-transition, and top-action reports.
- Diagnostics page showing output availability.
- Missing-output state from a clean checkout.

## Demo Boundaries

Do not describe the dashboard as production deployment. Do not claim CxT+, Contextual CxT, Advanced CxT, or OD-CxT are implemented. The v1 dashboard is a demo and portfolio surface over reproducible generated outputs.
