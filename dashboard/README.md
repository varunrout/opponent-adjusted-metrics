# CxG Analytics Dashboard

> Legacy/experimental dashboard path. The active dashboard entry point is
> `app/streamlit_app.py`, and the repository `make dashboard` target runs
> `poetry run streamlit run app/streamlit_app.py`. New promoted CxG portfolio
> views should be added to the active app, not to `dashboard/app.py`.

Interactive dashboard for Contextual Expected Goals (CxG) analysis.

## Features

- 📊 **Overview**: Model performance metrics and comparison with baseline
- ⚽ **Shot Explorer**: Interactive pitch maps with filters
- 📈 **Team Analysis**: League tables, finishing delta charts
- 👤 **Player Stats**: Individual player rankings and shot maps

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -r dashboard/requirements.txt

# Or with poetry (adds to existing environment)
poetry add streamlit mplsoccer plotly
```

### 2. Run the Dashboard

```bash
# From project root
streamlit run dashboard/app.py

# Or specify host/port
streamlit run dashboard/app.py --server.port 8501 --server.address localhost
```

### 3. Open in Browser

Navigate to `http://localhost:8501`

## Project Structure

```
dashboard/
├── app.py                      # Main entry point
├── requirements.txt            # Dashboard dependencies
├── README.md                   # This file
├── components/
│   ├── pitch_plots.py          # mplsoccer pitch visualizations
│   ├── charts.py               # Plotly interactive charts
│   └── data_loader.py          # Cached data loading utilities
└── pages/
    ├── 1_📊_Overview.py        # Model performance
    ├── 2_⚽_Shot_Explorer.py   # Interactive shot maps
    ├── 3_📈_Team_Analysis.py   # Team rankings
    └── 4_👤_Player_Stats.py    # Player analysis
```

## Data Requirements

The dashboard reads from `outputs/modeling/cxg/`:

- `cxg_dataset_enriched.parquet` - Main shot dataset
- `baseline_metrics.json` - Baseline model metrics
- `contextual_metrics_enriched.json` - CxG model metrics
- `contextual_feature_effects_enriched.csv` - Feature coefficients
- `prediction_runs/*/team_aggregates.csv` - Team totals

Make sure to run the modeling pipeline first to generate these files.

## Customization

### Adding New Pages

Create a new file in `pages/` with naming convention:
```
pages/5_🔧_New_Page.py
```

### Adding Team Names

Edit `components/data_loader.py` and update the `TEAM_NAMES` dictionary.

### Modifying Pitch Plots

All pitch visualizations use mplsoccer. See `components/pitch_plots.py` for customization.
