# oam-analyse: Gold (BigQuery oam_features) -> oam_analysis tables + GCS chart artifacts
# Build from repo root:  docker build -f docker/oam-analyse/Dockerfile -t oam-analyse .
FROM python:3.11-slim

WORKDIR /app

COPY docker/oam-analyse/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY scripts/materialize_cxg_defensive_involvement.py /app/scripts/materialize_cxg_defensive_involvement.py
COPY scripts/materialize_cxg_odi_features.py /app/scripts/materialize_cxg_odi_features.py
COPY scripts/materialize_cxg_defensive_profile_clusters.py /app/scripts/materialize_cxg_defensive_profile_clusters.py
COPY scripts/materialize_cxg_opponent_adjusted_analysis.py /app/scripts/materialize_cxg_opponent_adjusted_analysis.py
COPY scripts/register_chart_registry_for_run.py /app/scripts/register_chart_registry_for_run.py
COPY scripts/render_cxg_analysis_charts.py /app/scripts/render_cxg_analysis_charts.py
COPY scripts/render_cxg_feature_eda_appendix.py /app/scripts/render_cxg_feature_eda_appendix.py
COPY scripts/run_oam_analyse.py /app/scripts/run_oam_analyse.py

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["python", "scripts/run_oam_analyse.py"]
