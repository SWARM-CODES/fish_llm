#!/bin/bash

# Number of times to repeat the process
NUM_REPEATS=50

# Create output directory if it doesn't exist
#mkdir -p output

# Loop for the given number of repeats
for ((i=1; i<=NUM_REPEATS; i++))
do
    echo "Starting simulation run $i..."

    # Run the Python script and wait for it to complete
    python main.py

    # Check if simulation completed message is present
    if [ $? -eq 0 ]; then
        echo "Simulation run $i completed successfully."

        # Move and rename the output files
        mv iteration_1_explanations.json output_mechanism/output_onlyRheotaxis/iteration_${i}_explanations.json
        mv trajectories.nc output_mechanism/output_onlyRheotaxis/trajectories_${i}.nc

        echo "Files for run $i moved to output directory."
    else
        echo "Simulation run $i failed. Skipping file move."
    fi

    echo "--------------------------------------"
done

echo "All $NUM_REPEATS simulations completed."

