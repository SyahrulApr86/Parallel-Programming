#!/bin/bash
#SBATCH --job-name=heat-simulation
#SBATCH --partition=batch
#SBATCH --nodes=2-6          # Request between 2-6 nodes (flexible)
#SBATCH --ntasks-per-node=4
#SBATCH --output=heat_results_%j.out
#SBATCH --error=heat_errors_%j.err
#SBATCH --time=02:00:00

# Compile program
mpicc -o heat_diffusion heat_diffusion.c -lm

# Grid sizes to test
GRID_SIZES=(120 240 480 960)

# Iteration counts to test
ITERATIONS=(100 500 1000)

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
echo "Running experiments on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"

# Process counts to test (perfect squares for 2D grid)
PROCS=(4 9 16 25 36 49)

# Filter out process counts that exceed what we have available
AVAILABLE_PROCS=()
for np in "${PROCS[@]}"; do
    if [ $np -le $TOTAL_PROCS ]; then
        AVAILABLE_PROCS+=($np)
    fi
done

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

echo "=== Heat Diffusion with Different Process Counts ==="
echo "Grid size: 240x240, Iterations: 100"
echo "Format: (computation_time/communication_time) in seconds"
echo -n "NP\t"
for np in "${AVAILABLE_PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

echo -n "Time\t"
for np in "${AVAILABLE_PROCS[@]}"; do
    echo "Running heat diffusion with $np processes..." >&2
    
    # Run with timeout and capture result
    result=$(run_with_timeout "srun -n $np ./heat_diffusion 240 100")
    
    # Extract the result summary if it exists
    if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
        echo -n "$result\t"
    else
        summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
        if [ -z "$summary" ]; then
            echo -n "FAILED\t"
        else
            echo -n "$summary\t"
        fi
    fi
done
echo ""

echo ""
echo "=== Heat Diffusion with Different Grid Sizes ==="
echo "Process count: 16, Iterations: 100"
echo "Format: (computation_time/communication_time) in seconds"
echo -n "Size\t"
for size in "${GRID_SIZES[@]}"; do
    echo -n "${size}x${size}\t"
done
echo ""

# Check if we have enough processes for next tests
if [ 16 -le $TOTAL_PROCS ]; then
    echo -n "Time\t"
    for size in "${GRID_SIZES[@]}"; do
        echo "Running heat diffusion with grid size ${size}x${size}..." >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "srun -n 16 ./heat_diffusion $size 100")
        
        # Extract the result summary if it exists
        if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
            echo -n "$result\t"
        else
            summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$summary" ]; then
                echo -n "FAILED\t"
            else
                echo -n "$summary\t"
            fi
        fi
    done
else
    echo -n "Not enough processes available (need 16)"
fi
echo ""

echo ""
echo "=== Heat Diffusion with Different Iteration Counts ==="
echo "Grid size: 240x240, Process count: 16"
echo "Format: (computation_time/communication_time) in seconds"
echo -n "Iter\t"
for iter in "${ITERATIONS[@]}"; do
    echo -n "$iter\t"
done
echo ""

# Check if we have enough processes for next tests
if [ 16 -le $TOTAL_PROCS ]; then
    echo -n "Time\t"
    for iter in "${ITERATIONS[@]}"; do
        echo "Running heat diffusion with $iter iterations..." >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "srun -n 16 ./heat_diffusion 240 $iter")
        
        # Extract the result summary if it exists
        if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
            echo -n "$result\t"
        else
            summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$summary" ]; then
                echo -n "FAILED\t"
            else
                echo -n "$summary\t"
            fi
        fi
    done
else
    echo -n "Not enough processes available (need 16)"
fi
echo ""

# Experiment with reordering
echo ""
echo "=== Heat Diffusion with/without Reordering ==="
echo "Grid size: 240x240, Iterations: 100, Process count: 16"
echo -n "Reordering\t"
echo -n "With\t"
echo -n "Without"
echo ""

if [ 16 -le $TOTAL_PROCS ]; then
    echo -n "Time\t"
    
    # Test with reordering (default)
    result=$(run_with_timeout "srun -n 16 ./heat_diffusion 240 100")
    if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
        echo -n "$result\t"
    else
        summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
        if [ -z "$summary" ]; then
            echo -n "FAILED\t"
        else
            echo -n "$summary\t"
        fi
    fi
    
    # Create a version without reordering
    cp heat_diffusion.c heat_diffusion_no_reorder.c
    sed -i 's/reorder = 1;/reorder = 0;/' heat_diffusion_no_reorder.c
    mpicc -o heat_diffusion_no_reorder heat_diffusion_no_reorder.c -lm
    
    # Test without reordering
    result=$(run_with_timeout "srun -n 16 ./heat_diffusion_no_reorder 240 100")
    if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
        echo -n "$result"
    else
        summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
        if [ -z "$summary" ]; then
            echo -n "FAILED"
        else
            echo -n "$summary"
        fi
    fi
else
    echo -n "Not enough processes available (need 16)"
fi
echo ""

# Experiment with process distribution across nodes
echo ""
echo "=== Heat Diffusion with Different Process Distribution ==="
echo "Grid size: 240x240, Iterations: 100, Process count: 16"
echo -n "Distribution\t"
echo -n "Single Node\t"
echo -n "4 Nodes"
echo ""

if [ 16 -le $TOTAL_PROCS ]; then
    echo -n "Time\t"
    
    # Try to fit all processes on single node if possible
    if [ $SLURM_NTASKS_PER_NODE -ge 16 ]; then
        result=$(run_with_timeout "srun -n 16 -N 1 ./heat_diffusion 240 100")
        if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
            echo -n "$result\t"
        else
            summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$summary" ]; then
                echo -n "FAILED\t"
            else
                echo -n "$summary\t"
            fi
        fi
    else
        echo -n "N/A\t"
    fi
    
    # Distribute across 4 nodes if available
    if [ $SLURM_JOB_NUM_NODES -ge 4 ]; then
        result=$(run_with_timeout "srun -n 16 -N 4 ./heat_diffusion 240 100")
        if [[ $result == TIMEOUT* ]] || [[ $result == ERROR* ]]; then
            echo -n "$result"
        else
            summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$summary" ]; then
                echo -n "FAILED"
            else
                echo -n "$summary"
            fi
        fi
    else
        echo -n "N/A"
    fi
else
    echo -n "Not enough processes available (need 16)"
fi
echo ""

echo "All tests completed"