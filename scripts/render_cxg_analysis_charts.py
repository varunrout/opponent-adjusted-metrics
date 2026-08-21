"""CLI wrapper for CxG analysis chart rendering."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from opponent_adjusted.analysis.cxg_charts import main


if __name__ == "__main__":
    main()
