"""Contract-driven data loading for the v1 Streamlit dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "configs" / "dashboard" / "v1_dashboard_contract.json"


@dataclass(frozen=True)
class ResourceStatus:
    metric: str
    name: str
    path: str
    found: bool
    missing: bool
    row_count: int
    column_count: int
    error: str | None = None


@dataclass(frozen=True)
class DashboardResource:
    metric: str
    name: str
    spec: dict[str, Any]
    status: ResourceStatus
    dataframe: pd.DataFrame
    json_data: dict[str, Any]


def load_dashboard_contract(contract_path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load the dashboard data contract."""

    return json.loads(contract_path.read_text(encoding="utf-8"))


def iter_resource_specs(contract: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Flatten contract metric inputs into `(metric, input_name, spec)` tuples."""

    resources = []
    for metric, section in contract.get("metric_sections", {}).items():
        for input_name, spec in section.get("inputs", {}).items():
            resources.append((metric, input_name, spec))
    return resources


def expected_columns(spec: dict[str, Any]) -> list[str]:
    """Return required plus optional columns for a tabular dashboard input."""

    columns = [*spec.get("required_columns", []), *spec.get("optional_columns", [])]
    return list(dict.fromkeys(columns))


def _empty_dataframe(spec: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(columns=expected_columns(spec))


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root / path


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported dashboard data format: {path.suffix}")


def load_resource(
    metric: str,
    name: str,
    spec: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> DashboardResource:
    """Load a dashboard resource from its contract spec.

    Missing files return an empty DataFrame with expected columns and a status
    record instead of raising. JSON report files return `json_data` when found.
    """

    contract_path = spec.get("path", "")
    resolved_path = _resolve_path(contract_path, project_root)
    is_json_report = resolved_path.suffix.lower() == ".json" and "required_fields" in spec

    if not resolved_path.exists():
        dataframe = _empty_dataframe(spec)
        status = ResourceStatus(
            metric=metric,
            name=name,
            path=str(resolved_path),
            found=False,
            missing=True,
            row_count=0,
            column_count=len(dataframe.columns),
        )
        return DashboardResource(metric, name, spec, status, dataframe, {})

    try:
        if is_json_report:
            json_data = json.loads(resolved_path.read_text(encoding="utf-8"))
            dataframe = pd.DataFrame([json_data]) if json_data else pd.DataFrame()
        else:
            json_data = {}
            dataframe = _read_tabular(resolved_path)
        status = ResourceStatus(
            metric=metric,
            name=name,
            path=str(resolved_path),
            found=True,
            missing=False,
            row_count=int(len(dataframe)),
            column_count=int(len(dataframe.columns)),
        )
        return DashboardResource(metric, name, spec, status, dataframe, json_data)
    except Exception as exc:
        dataframe = _empty_dataframe(spec)
        status = ResourceStatus(
            metric=metric,
            name=name,
            path=str(resolved_path),
            found=False,
            missing=True,
            row_count=0,
            column_count=len(dataframe.columns),
            error=str(exc),
        )
        return DashboardResource(metric, name, spec, status, dataframe, {})


def load_all_resources(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, DashboardResource]]:
    """Load all resources declared in the dashboard contract."""

    contract = load_dashboard_contract(contract_path)
    loaded: dict[str, dict[str, DashboardResource]] = {}
    for metric, name, spec in iter_resource_specs(contract):
        loaded.setdefault(metric, {})[name] = load_resource(
            metric, name, spec, project_root=project_root
        )
    return loaded


def status_table(resources: dict[str, dict[str, DashboardResource]]) -> pd.DataFrame:
    """Build a status table for dashboard display and tests."""

    rows = []
    for metric_resources in resources.values():
        for resource in metric_resources.values():
            rows.append(
                {
                    "metric": resource.status.metric,
                    "name": resource.status.name,
                    "found": resource.status.found,
                    "missing": resource.status.missing,
                    "path": resource.status.path,
                    "row_count": resource.status.row_count,
                    "column_count": resource.status.column_count,
                    "error": resource.status.error,
                }
            )
    return pd.DataFrame(rows)
