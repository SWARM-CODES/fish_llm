#!/usr/bin/env bash
set -euo pipefail

NUM_REPEATS=12
OUTDIR="output"
#mkdir -p "$OUTDIR"

for i in $(seq 8 "$NUM_REPEATS"); do
  echo "======================================"
  echo "Starting simulation run $i..."

  # Run Python script (synchronously)
  if python3 main.py > test.log 2>&1; then
    echo "Simulation run $i completed successfully."

    # Move and rename the output files
    if [[ -f trajectories.nc ]]; then
      mv -f trajectories.nc "$OUTDIR/trajectories${i}.nc"
    else
      echo "Warning: trajectories.nc not found for run $i."
    fi

    if [[ -f test.log ]]; then
      mv -f test.log "$OUTDIR/record${i}.log"
    fi

    echo "Files for run $i moved to output directory."
  else
    status=$?
    echo "Simulation run $i failed with status $status. Skipping file move."
  fi

  echo "--------------------------------------"
done

echo "All $NUM_REPEATS simulations completed."

