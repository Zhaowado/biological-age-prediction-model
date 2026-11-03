#!/bin/sh
set -eu
python3 calculate_ba.py examples/example_input.csv output/example_output.csv
python3 -m unittest discover -s tests -v
echo "Example output: output/example_output.csv"
