from pathlib import Path

import pandas as pd

from opponent_adjusted.analysis.cxg.v1.report import run_cxg_analysis
from tests.test_analysis_shared_loaders import _create_cxg_fixture_db


def test_cxg_analysis_writes_expected_outputs(tmp_path: Path):
    db_path = _create_cxg_fixture_db(tmp_path / "fixture.db")
    output_dir = tmp_path / "analysis" / "cxg"

    result = run_cxg_analysis(output_dir=output_dir, db_path=db_path)

    assert result["model_version"] == "cxg_latest"
    assert result["shot_count"] == 2
    expected_paths = [
        output_dir / "eda" / "tables" / "shot_population_summary.csv",
        output_dir / "eda" / "tables" / "shot_outcome_summary.csv",
        output_dir / "distributions" / "tables" / "cxg_distribution_summary.csv",
        output_dir / "distributions" / "tables" / "opponent_adjustment_summary.csv",
        output_dir / "distributions" / "plots" / "cxg_distribution.png",
        output_dir / "distributions" / "plots" / "opponent_adjustment_distribution.png",
        output_dir / "slices" / "tables" / "by_body_part.csv",
        output_dir / "slices" / "tables" / "by_pressure.csv",
        output_dir / "slices" / "tables" / "by_minute_bucket.csv",
        output_dir / "slices" / "tables" / "by_opponent.csv",
        output_dir / "players" / "tables" / "top_players_by_cxg.csv",
        output_dir / "players" / "tables" / "shot_quality_vs_volume.csv",
        output_dir / "players" / "plots" / "player_shot_quality_vs_volume.png",
        output_dir / "teams" / "tables" / "top_teams_by_cxg.csv",
        output_dir / "teams" / "tables" / "team_quality_vs_volume.csv",
        output_dir / "teams" / "plots" / "team_shot_quality_vs_volume.png",
        output_dir / "report.md",
    ]
    for path in expected_paths:
        assert path.exists(), path

    population = pd.read_csv(output_dir / "eda" / "tables" / "shot_population_summary.csv")
    assert int(population.loc[0, "shot_count"]) == 2
    assert float(population.loc[0, "goal_rate"]) == 0.5
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Goal rate" in report
    assert "Skipped Optional Slices" in report


def test_cxg_analysis_skips_missing_optional_slices(tmp_path: Path):
    db_path = _create_cxg_fixture_db(tmp_path / "fixture.db", include_optional=False)
    output_dir = tmp_path / "analysis" / "cxg"

    result = run_cxg_analysis(output_dir=output_dir, db_path=db_path)

    assert "by_pressure" in result["skipped_slices"]
    assert "by_minute_bucket" in result["skipped_slices"]
    assert not (output_dir / "slices" / "tables" / "by_pressure.csv").exists()
    assert (output_dir / "slices" / "tables" / "by_body_part.csv").exists()
    assert (output_dir / "report.md").exists()


def test_makefile_exposes_cxg_analysis_target():
    text = Path("Makefile").read_text(encoding="utf-8")

    assert "analysis-cxg:" in text
    assert "poetry run python scripts/run_cxg_analysis_v1.py" in text
    assert "analysis-v1: analysis-cxg" in text
