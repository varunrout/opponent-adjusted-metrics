"""Local dev entrypoint for the FastAPI backend (used by .claude/launch.json's
"api-dev" config). Inserts `src` onto `sys.path` itself rather than relying on
the caller to set `PYTHONPATH`, since the package isn't installed in the active
Python environment -- this matches how `pytest` already resolves imports here
(via its own rootdir config), just made explicit for a plain `uvicorn` run."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("opponent_adjusted.api.main:app", host="0.0.0.0", port=8000)
