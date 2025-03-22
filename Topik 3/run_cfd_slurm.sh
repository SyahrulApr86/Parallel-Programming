#!/bin/bash
#SBATCH --job-name=cfd-simulation
#SBATCH --partition=batch
#SBATCH --nodes=2-6          # Request between 2-6 nodes (flexible)
#SBATCH --output=cfd_results_%j.out
#SBATCH --error=cfd_errors_%j.err

# Compile program
mpicc -o cfd_simulation cfd_simulation.c -lm

# Domain sizes to test (width x height)
DOMAIN_SIZES=("100 50" "200 100" "400 100" "800 200")

# Reynolds numbers to test
REYNOLDS=(10 100 500 1000)

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
echo "Running experiments on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"

# Process counts to test (perfect squares for 2D grid)
PROCS=(4 9 16)
if [ $TOTAL_PROCS -ge 25 ]; then
    PROCS+=(25)
fi
if [ $TOTAL_PROCS -ge 36 ]; then
    PROCS+=(36)
fi
if [ $TOTAL_PROCS -ge 49 ]; then
    PROCS+=(49)
fi

# Function to run a command with timeout
run_with_timeout() {
    local cmd="$1"
    local timeout=600  # 10 minutes timeout
    
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

echo "=== CFD Simulation with Different Process Counts ==="
echo "Format: (computation_time/communication_time) in seconds"
echo "Domain: 400x100, Reynolds: 100"
echo -n "NP\t"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

echo -n "Time\t"
for np in "${PROCS[@]}"; do
    # Check if process count is available
    if [ $np -le $TOTAL_PROCS ]; then
        echo "Running CFD with np=$np" >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "srun -n $np ./cfd_simulation 400 100 100")
        
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
    else
        echo -n "N/A\t"
    fi
done
echo ""

echo ""
echo "=== CFD Simulation with Different Domain Sizes ==="
echo "Format: (computation_time/communication_time) in seconds"
echo "Process count: 16, Reynolds: 100"
echo -n "Domain\t"
for domain in "${DOMAIN_SIZES[@]}"; do
    echo -n "$domain\t"
done
echo ""

echo -n "Time\t"
# Check if we have enough processes for all tests
if [ 16 -le $TOTAL_PROCS ]; then
    for domain in "${DOMAIN_SIZES[@]}"; do
        nx=$(echo $domain | cut -d' ' -f1)
        ny=$(echo $domain | cut -d' ' -f2)
        
        echo "Running CFD with domain=$domain" >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "srun -n 16 ./cfd_simulation $nx $ny 100")
        
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
    echo -n "Not enough processes (need 16)"
fi
echo ""

echo ""
echo "=== CFD Simulation with Different Reynolds Numbers ==="
echo "Format: (computation_time/communication_time) in seconds"
echo "Domain: 400x100, Process count: 16"
echo -n "Re\t"
for re in "${REYNOLDS[@]}"; do
    echo -n "$re\t"
done
echo ""

echo -n "Time\t"
# Check if we have enough processes for all tests
if [ 16 -le $TOTAL_PROCS ]; then
    for re in "${REYNOLDS[@]}"; do
        echo "Running CFD with Reynolds=$re" >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "srun -n 16 ./cfd_simulation 400 100 $re")
        
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
    echo -n "Not enough processes (need 16)"
fi
echo ""

# Experiment with reordering
echo ""
echo "=== CFD Simulation with/without Reordering ==="
echo "Format: (computation_time/communication_time) in seconds"
echo "Domain: 400x100, Reynolds: 100, Process count: 16"
echo -n "Reordering\t"
echo -n "With\t"
echo -n "Without"
echo ""

echo -n "Time\t"
if [ 16 -le $TOTAL_PROCS ]; then
    # Test with reordering (default)
    result=$(run_with_timeout "srun -n 16 ./cfd_simulation 400 100 100")
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
    cp cfd_simulation.c cfd_simulation_no_reorder.c
    sed -i 's/reorder = 1;/reorder = 0;/' cfd_simulation_no_reorder.c
    mpicc -o cfd_simulation_no_reorder cfd_simulation_no_reorder.c -lm
    
    # Test without reordering
    result=$(run_with_timeout "srun -n 16 ./cfd_simulation_no_reorder 400 100 100")
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
    echo -n "Not enough processes (need 16)"
fi
echo ""

# Experiment with process distribution across nodes
echo ""
echo "=== CFD Simulation with Different Process Distribution ==="
echo "Format: (computation_time/communication_time) in seconds"
echo "Domain: 400x100, Reynolds: 100, Process count: 16"
echo -n "Distribution\t"
echo -n "Single Node\t"
echo -n "4 Nodes"
echo ""

echo -n "Time\t"
if [ 16 -le $TOTAL_PROCS ]; then
    # Try to fit all processes on single node if possible
    if [ $SLURM_NTASKS_PER_NODE -ge 16 ]; then
        result=$(run_with_timeout "srun -n 16 -N 1 ./cfd_simulation 400 100 100")
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
        result=$(run_with_timeout "srun -n 16 -N 4 ./cfd_simulation 400 100 100")
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
    echo -n "Not enough processes (need 16)"
fi
echo ""

echo "All tests completed"