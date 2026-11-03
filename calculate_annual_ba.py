#!/usr/bin/env python3
"""Aggregate repeated visits by year and create long and wide BA CSV files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fitness_ba.annual import read_and_score


def main():
    parser = argparse.ArgumentParser(description="Calculate one BA per person-year")
    parser.add_argument("input_csv", help="wide visits or item-value long CSV")
    parser.add_argument("output_wide_csv")
    parser.add_argument("--basic-info", help="optional oneid/sex/birthday CSV for item-value long input")
    parser.add_argument("--output-long", help="optional detailed person-year output")
    args = parser.parse_args()
    long, wide = read_and_score(args.input_csv, args.basic_info)
    wide_target = Path(args.output_wide_csv)
    wide_target.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_target, index=False, encoding="utf-8-sig", float_format="%.6f")
    if args.output_long:
        target = Path(args.output_long)
        target.parent.mkdir(parents=True, exist_ok=True)
        for column in long.columns:
            if long[column].map(lambda value: isinstance(value, (dict, list))).any():
                long[column] = long[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        long.to_csv(target, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"Scored {len(long)} person-years for {long['oneid'].nunique()} people -> {wide_target}")


if __name__ == "__main__":
    main()
