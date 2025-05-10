#!/bin/bash
# Script to run process topology experiments on multicore system

# Compile the program
mpicc -o topology_test topology_test.c -lm

# Output file
OUTPUT_FILE="topology_results_multicore_$(date +%s).txt"

# Process counts to test (perfect squares for 2D grids)
PROCS=(1 4 9 16 25 36 49 64)

# Write header
echo "Process Topology Experiments on Multicore System" | tee -a "$OUTPUT_FILE"
echo "================================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Run all tests for different process counts
for np in "${PROCS[@]}"; do
    echo "Running tests with $np processes" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------" | tee -a "$OUTPUT_FILE"
    
    # Test 1: Cartesian Topology Creation
    echo "Test 1: Cartesian Topology Creation with $np processes" | tee -a "$OUTPUT_FILE"
    mpirun -np $np ./topology_test 1 2>&1 | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    # Test 2: Neighbor Communication
    echo "Test 2: Neighbor Communication with $np processes" | tee -a "$OUTPUT_FILE"
    mpirun -np $np ./topology_test 2 2>&1 | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    # Test 3: Cart_sub (only for np >= 8)
    if [ $np -ge 8 ]; then
        echo "Test 3: Cart_sub with $np processes" | tee -a "$OUTPUT_FILE"
        mpirun -np $np ./topology_test 3 2>&1 | tee -a "$OUTPUT_FILE"
        echo "" | tee -a "$OUTPUT_FILE"
    else
        echo "Test 3: Cart_sub - Skipped (requires at least 8 processes)" | tee -a "$OUTPUT_FILE"
        echo "" | tee -a "$OUTPUT_FILE"
    fi
    
    # Test 4: 5-point Stencil Communication
    echo "Test 4: 5-point Stencil Communication with $np processes" | tee -a "$OUTPUT_FILE"
    mpirun -np $np ./topology_test 4 2>&1 | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "All tests completed. Results saved to $OUTPUT_FILE"