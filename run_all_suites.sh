#!/bin/bash

cd "$(dirname "$0")/webserver"

python ../benchmark/cli.py --suite ai-agents -t 50 --seed 42 --port 8123 --raw-data --format csv -k test

python ../benchmark/cli.py --suite network-latency -t 50 --seed 42 --port 8123 --raw-data --format csv -k test

python ../benchmark/cli.py --suite image-size -t 50 --seed 42 --port 8123 --raw-data --format csv -k test

python ../benchmark/cli.py --suite scalability -t 50 --seed 42 --port 8123 --raw-data --format csv -k test

# Zip all merged CSV files
cd ../benchmark/results
zip merged_results.zip *_merged.csv
echo "Merged CSV files zipped to benchmark/results/merged_results.zip"