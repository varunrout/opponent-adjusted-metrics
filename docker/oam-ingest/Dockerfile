# oam-ingest: raw StatsBomb open data -> Bronze (GCS)
# Build from repo root:  docker build -f docker/oam-ingest/Dockerfile -t oam-ingest .
FROM python:3.11-slim

WORKDIR /app

COPY docker/oam-ingest/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY scripts/fetch_statsbomb_subset.py /app/scripts/fetch_statsbomb_subset.py
COPY configs/ /app/configs/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/fetch_statsbomb_subset.py"]
