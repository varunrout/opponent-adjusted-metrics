#!/usr/bin/env python
"""Run the DB-backed CxG v1 football analysis report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opponent_adjusted.analysis.cxg.v1.report import run_cxg_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CxG v1 analysis report")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cxg_analysis(output_dir=args.output_dir, db_path=args.db_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
