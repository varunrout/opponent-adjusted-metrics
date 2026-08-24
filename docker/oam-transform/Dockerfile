# oam-transform: Bronze (GCS) -> Silver (GCS) -> oam_core + oam_features (BigQuery)
# Build from repo root:  docker build -f docker/oam-transform/Dockerfile -t oam-transform .
FROM python:3.11-slim

WORKDIR /app

# build_statsbomb_silver.py stamps GIT_COMMIT_SHA into the run manifest via
# `git rev-parse HEAD` (scripts/build_statsbomb_silver.py::_git_sha, unmodified) -- needs
# both the git binary and a resolvable commit (found missing during real end-to-end
# validation: python:3.11-slim has neither by default). The repo's real .git lives outside
# this build context (this checkout is a git worktree pointing at a sibling repo), so it
# can't be COPY'd in portably. Synthesizing a single-commit repo at build time is a
# documented, deliberate tradeoff: `git rev-parse HEAD` resolves to a valid 40-char SHA
# (this build's own synthetic commit, not the source repo's actual commit) so the script
# doesn't crash, rather than papering over the crash by editing the script itself.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && git init -q \
    && git config user.email "ci@oam.local" \
    && git config user.name "oam-container-build" \
    && git commit -q --allow-empty -m "oam-transform container build"

COPY docker/oam-transform/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/build_statsbomb_silver.py /app/scripts/build_statsbomb_silver.py
COPY scripts/publish_oam_core.py /app/scripts/publish_oam_core.py
COPY scripts/materialize_gold_cxg.py /app/scripts/materialize_gold_cxg.py
COPY scripts/materialize_cxg_feature_family_tables.py /app/scripts/materialize_cxg_feature_family_tables.py
COPY scripts/audit_cxg_e13_f1_f15.py /app/scripts/audit_cxg_e13_f1_f15.py
COPY scripts/run_oam_transform.py /app/scripts/run_oam_transform.py

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/run_oam_transform.py"]
