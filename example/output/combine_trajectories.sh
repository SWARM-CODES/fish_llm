#!/bin/bash

# Define output filename
OUTPUT_FILE="trajectories_combined.nc"

# Check if NCO is installed
#if ! command -v ncrcat &> /dev/null; then
#    echo "Error: NCO (NetCDF Operators) is not installed. Install it using: sudo apt install nco"
#    exit 1
#fi

# Combine files along the particles dimension using ncrcat
echo "Combining NetCDF files along the particles dimension..."
ncrcat -v x,y,z,bathy,temperature trajectories_{1..50}.nc -o $OUTPUT_FILE

# Check if the process was successful
if [ $? -eq 0 ]; then
    echo "Combination completed successfully. Output saved as $OUTPUT_FILE"
else
    echo "Error: Failed to combine NetCDF files."
#    exit 1
fi

