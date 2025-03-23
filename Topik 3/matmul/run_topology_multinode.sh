#!/bin/bash
#SBATCH --job-name=mpi-topology
#SBATCH --partition=batch
#SBATCH --nodes=2-6          # Request between 2-6 nodes (flexible)
#SBATCH --output=topology_results_%j.out
#SBATCH --error=topology_errors_%j.err

# Compile all programs
echo "Compiling programs..."
mpicc -o matvec matvec_fixed.c -lm
mpicc -o matvec_cart matvec_cart.c -lm
mpicc -o cannon_cart cannon_cart.c -lm

# Matrix sizes to test
SIZES=(128 256 512 1024 2048)

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * 8))  # Assuming 8 cores per node
echo "Running experiments on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"

# Process counts to test
PROCS=(1 2 4 8 16)
if [ $TOTAL_PROCS -ge 32 ]; then
    PROCS+=(32)
fi
if [ $TOTAL_PROCS -ge 48 ]; then
    PROCS+=(48)
fi

# Generate square process counts for Cannon
SQUARE_PROCS=()
for (( i=1; i*i <= $TOTAL_PROCS; i++ )); do
    SQUARE_PROCS+=($((i*i)))
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

echo "===== MPI TOPOLOGY EXPERIMENTS ====="
echo "Hostname: $(hostname)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Total processes: $TOTAL_PROCS"
echo ""

echo "=== Matrix-Vector Multiplication: Standard vs Cartesian Topology ==="
echo "Format: (computation_time/communication_time) in seconds"
echo -n "| N | NP | Standard | Cartesian |"
echo -e "\n|---|---|----------|-----------|"

for n in "${SIZES[@]}"; do
    for np in "${PROCS[@]}"; do
        # Check if matrix size is divisible by process count
        if [ $(($n % $np)) -ne 0 ]; then
            echo "| $n | $np | N/A | N/A |"
            continue
        fi
        
        echo "Running matrix-vector tests with n=$n, np=$np" >&2
        
        # Run standard implementation
        std_result=$(run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./matvec $n")
        if [[ $std_result == TIMEOUT* ]] || [[ $std_result == ERROR* ]]; then
            std_summary="$std_result"
        else
            std_summary=$(echo "$std_result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$std_summary" ]; then
                std_summary="FAILED"
            fi
        fi
        
        # Run Cartesian topology implementation
        cart_result=$(run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./matvec_cart $n")
        if [[ $cart_result == TIMEOUT* ]] || [[ $cart_result == ERROR* ]]; then
            cart_summary="$cart_result"
        else
            cart_summary=$(echo "$cart_result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$cart_summary" ]; then
                cart_summary="FAILED"
            fi
        fi
        
        echo "| $n | $np | $std_summary | $cart_summary |"
    done
done

echo ""
echo "=== Cannon's Algorithm with 2D Cartesian Topology ==="
echo "Format: (computation_time/communication_time) in seconds"
echo -n "| N | NP | Cannon-Cart |"
echo -e "\n|---|---|-------------|"

for n in "${SIZES[@]}"; do
    for np in "${SQUARE_PROCS[@]}"; do
        # Calculate square root of np
        sqrt_np=$(echo "sqrt($np)" | bc)
        
        # Check if matrix size is divisible by sqrt(np)
        if [ $(($n % $sqrt_np)) -ne 0 ]; then
            # Adjust matrix size to be divisible
            adjusted_n=$(( ($n / $sqrt_np) * $sqrt_np ))
            echo "Running Cannon with adjusted n=$adjusted_n (was $n), np=$np" >&2
            n_display="$n ($adjusted_n)"
        else
            echo "Running Cannon with n=$n, np=$np" >&2
            adjusted_n=$n
            n_display=$n
        fi
        
        # Run Cannon with Cartesian topology
        cannon_result=$(run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./cannon_cart $adjusted_n")
        if [[ $cannon_result == TIMEOUT* ]] || [[ $cannon_result == ERROR* ]]; then
            cannon_summary="$cannon_result"
        else
            cannon_summary=$(echo "$cannon_result" | grep "Result summary" | awk '{print $3}')
            if [ -z "$cannon_summary" ]; then
                cannon_summary="FAILED"
            fi
        fi
        
        echo "| $n_display | $np | $cannon_summary |"
    done
done

echo ""
echo "All tests completed"