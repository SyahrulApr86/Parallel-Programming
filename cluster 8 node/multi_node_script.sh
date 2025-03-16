#!/bin/bash
#SBATCH --job-name=matrix-mult
#SBATCH --partition=batch
#SBATCH --nodes=2-6          # Request between 2-6 nodes (flexible)
#SBATCH --output=results_%j.out
#SBATCH --error=errors_%j.err

# Compile programs
mpicc -o matvec matvec_fixed.c -lm
mpicc -o matmat matmat_fixed.c -lm
mpicc -o cannon cannon_fixed.c -lm

# Matrix sizes to test
SIZES=(128 256 512 1024 2048)

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * 8))
echo "Running experiments on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"

# Process counts to test
PROCS=(1 2 4 8 16)
if [ $TOTAL_PROCS -ge 32 ]; then
    PROCS+=(32)
fi
if [ $TOTAL_PROCS -ge 48 ]; then
    PROCS+=(48)
fi

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

echo "=== Matrix-Vector Multiplication Experiments ==="
echo "Format: (computation_time/communication_time) in seconds"
echo -n "N\t"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

for n in "${SIZES[@]}"; do
    echo -n "$n\t"
    for np in "${PROCS[@]}"; do
        # Check if matrix size is divisible by process count
        if [ $(($n % $np)) -ne 0 ]; then
            echo -n "N/A\t"
            continue
        fi
        
        echo "Running matvec with n=$n, np=$np" >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./matvec $n")
        
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
        
        # Flush output
        echo -n "" >&2
    done
    echo ""
done

echo ""
echo "=== Matrix-Matrix Multiplication Experiments ==="
echo "Format: (computation_time/communication_time) in seconds"
echo -n "N\t"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

for n in "${SIZES[@]}"; do
    echo -n "$n\t"
    for np in "${PROCS[@]}"; do
        # Check if matrix size is divisible by process count
        if [ $(($n % $np)) -ne 0 ]; then
            echo -n "N/A\t"
            continue
        fi
        
        echo "Running matmat with n=$n, np=$np" >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./matmat $n")
        
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
        
        # Flush output
        echo -n "" >&2
    done
    echo ""
done

# Cannon with perfect square process counts
echo ""
echo "=== Cannon's Algorithm Matrix Multiplication Experiments ==="
echo "Format: (computation_time/communication_time) in seconds"
echo -n "N\t"

# Select perfect square process counts based on availability
SQUARE_PROCS=(1 4 9 16)
if [ $TOTAL_PROCS -ge 25 ]; then
    SQUARE_PROCS+=(25)
fi
if [ $TOTAL_PROCS -ge 36 ]; then
    SQUARE_PROCS+=(36)
fi
if [ $TOTAL_PROCS -ge 49 ]; then
    SQUARE_PROCS+=(49)
fi

for np in "${SQUARE_PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

for n in "${SIZES[@]}"; do
    echo -n "$n\t"
    for np in "${SQUARE_PROCS[@]}"; do
        # Calculate square root of np
        sqrt_np=$(echo "sqrt($np)" | bc)
        
        # Check if matrix size is divisible by sqrt(np)
        if [ $(($n % $sqrt_np)) -ne 0 ]; then
            # Adjust matrix size to be divisible
            adjusted_n=$(( ($n / $sqrt_np) * $sqrt_np ))
            echo -n "($adjusted_n) "
        else
            adjusted_n=$n
        fi
        
        echo "Running cannon with n=$adjusted_n, np=$np" >&2
        
        # Run with timeout and capture result
        result=$(run_with_timeout "mpirun --mca btl_base_warn_component_unused 0 -np $np ./cannon $adjusted_n")
        
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
        
        # Flush output
        echo -n "" >&2
    done
    echo ""
done

echo "All tests completed"