#!/usr/bin/env python3
"""Batch score one assessment per CSV row."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fitness_ba import score_dataframe


def main():
    parser = argparse.ArgumentParser(description="Calculate FAI and exploratory BA from fitness scores")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    data = pd.read_csv(args.input_csv, dtype={"oneid": str})
    output = score_dataframe(data)
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (dict, list))).any():
            output[column] = output[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
    target = Path(args.output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"Scored {len(output)} rows -> {target}")


if __name__ == "__main__":
    main()
