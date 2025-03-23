#!/bin/bash
#SBATCH --job-name=topo-test
#SBATCH --partition=batch
#SBATCH --nodes=1-8          # Request between 1-8 nodes (flexible)
#SBATCH --output=topo_results_%j.out
#SBATCH --error=topo_errors_%j.err

# Compile the program
mpicc -o topology_test topology_test.c -lm

# Process counts to test (perfect squares for 2D grids)
PROCS=(1 4 9 16 25 36 49 64)

# Write header
echo "Process Topology Experiments on Cluster Environment"
echo "=================================================="
echo ""

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * 8))
echo "Running on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"
echo ""

# Function to run a command with timeout
run_with_timeout() {
    local cmd="$1"
    local timeout=300  # 5 minutes timeout
    
    # Create a temporary file for error output
    local errfile=$(mktemp)
    
    # Run the command with timeout
    timeout --kill-after=30s $timeout bash -c "$cmd" 2>$errfile
    local exit_code=$?
    
    # Check for timeout or error
    if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "TIMEOUT"
        echo "Command timed out after ${timeout}s: $cmd" >&2
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR($exit_code)"
        echo "Command failed with exit code $exit_code: $cmd" >&2
        cat $errfile >&2
    fi
    
    # Clean up
    rm -f $errfile
}

# Run only process counts that fit within available resources
for np in "${PROCS[@]}"; do
    if [ $np -le $TOTAL_PROCS ]; then
        echo "Running tests with $np processes"
        echo "----------------------------------"
        
        # Test 1: Cartesian Topology Creation
        echo "Test 1: Cartesian Topology Creation with $np processes"
        run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./topology_test 1"
        echo ""
        
        # Test 2: Neighbor Communication
        echo "Test 2: Neighbor Communication with $np processes"
        run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./topology_test 2"
        echo ""
        
        # Test 3: Cart_sub (only for np >= 8)
        if [ $np -ge 8 ]; then
            echo "Test 3: Cart_sub with $np processes"
            run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./topology_test 3"
            echo ""
        else
            echo "Test 3: Cart_sub - Skipped (requires at least 8 processes)"
            echo ""
        fi
        
        # Test 4: 5-point Stencil Communication
        echo "Test 4: 5-point Stencil Communication with $np processes"
        run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./topology_test 4"
        echo ""
        
        echo ""
    fi
done

echo "All tests completed."