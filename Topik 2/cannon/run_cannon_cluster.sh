#!/bin/bash
#SBATCH --job-name=cannon-cluster
#SBATCH --partition=batch
#SBATCH --nodes=1-8        # Request up to 8 nodes
#SBATCH --output=cannon_cluster_%j.out
#SBATCH --error=cannon_cluster_%j.err
#SBATCH --time=24:00:00    # Set maximum run time to 24 hours

# Compile the program
mpicc -o cannon cannon.c -lm -O3

# Matrix sizes to test (must be divisible by all grid sizes)
SIZES=(480 960 1920 3840)

# Process counts to test - must be perfect squares for Cannon's algorithm
PROCS=(1 4 9 16 25 36 49 64)

# Create results directory
RESULTS_DIR="cannon_results_cluster"
mkdir -p "$RESULTS_DIR"

# Create CSV file for results
RESULTS_CSV="$RESULTS_DIR/cannon_results_cluster.csv"
echo "Size,np,ComputeTime,CommunicationTime,TotalTime" > "$RESULTS_CSV"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Cannon's Algorithm - Cluster Environment Results" > "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Environment A (8 Nodes Cluster)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Format: ComputeTime/CommunicationTime (seconds)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * 8))
echo "Running on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"
echo "" | tee -a "$SUMMARY_FILE"

# Function to run a command with timeout
run_with_timeout() {
    local cmd="$1"
    local timeout=10800  # 3 hour timeout for each test case
    
    # Create a temporary file for output
    local outfile=$(mktemp)
    
    # Print the command we're running (to stderr)
    echo "RUNNING: $cmd" >&2
    
    # Run the command with timeout
    timeout --kill-after=60s $timeout bash -c "$cmd" > "$outfile" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "TIMEOUT" >&2
        echo "Command timed out after ${timeout}s: $cmd" >&2
        echo "TIMEOUT/TIMEOUT"  # Return format for summary table
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR($exit_code)" >&2
        echo "Command failed with exit code $exit_code: $cmd" >&2
        cat "$outfile" >&2  # Print output to help debug
        echo "ERROR/ERROR"  # Return format for summary table
    else
        # Extract results from the output file
        local compute_time=$(grep "Computation time:" "$outfile" | awk '{print $3}')
        local comm_time=$(grep "Communication time:" "$outfile" | awk '{print $3}')
        local total_time=$(grep "Total time:" "$outfile" | awk '{print $3}')
        
        # If we can't find those specific labels, try to match the result summary line format
        if [ -z "$compute_time" ] || [ -z "$comm_time" ]; then
            local result_line=$(grep "Result summary" "$outfile")
            if [ ! -z "$result_line" ]; then
                # Extract values from parentheses (x.xxxx/y.yyyy)
                result_values=$(echo "$result_line" | grep -o '([^)]*)')
                compute_time=$(echo "$result_values" | sed -E 's/\(([0-9.]+)\/([0-9.]+)\)/\1/')
                comm_time=$(echo "$result_values" | sed -E 's/\(([0-9.]+)\/([0-9.]+)\)/\2/')
            fi
        fi
        
        echo "$compute_time/$comm_time"  # Return format for summary table
        
        # Add to CSV
        echo "$N,$np,$compute_time,$comm_time,$total_time" >> "$RESULTS_CSV"
    fi
    
    # Copy the output to the result file
    cp "$outfile" "$OUTPUT_FILE"
    
    # Clean up
    rm -f "$outfile"
}

# Table header
echo -n "      |" >> "$SUMMARY_FILE"
for np in "${PROCS[@]}"; do
    printf " np=%-4s |" "$np" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"

echo -n "------|" >> "$SUMMARY_FILE"
for np in "${PROCS[@]}"; do
    echo -n "----------|" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"

# Run tests for each matrix size and process count
for N in "${SIZES[@]}"; do
    echo "====================================================="
    echo "Running Cannon's Algorithm with N=$N"
    echo "====================================================="
    
    # Start row in summary table
    printf "N=%-6s|" "$N" >> "$SUMMARY_FILE"
    
    for np in "${PROCS[@]}"; do
        # Skip if the number of processes is greater than available
        if [ $np -gt $TOTAL_PROCS ]; then
            echo "Skipping np=$np (exceeds available processes)" >&2
            printf " N/A      |" >> "$SUMMARY_FILE"
            continue
        fi
        
        # Calculate grid size (sqrt of number of processes)
        grid_size=$(echo "sqrt($np)" | bc)
        
        # Skip if matrix size is not divisible by grid size
        if [ $((N % grid_size)) -ne 0 ]; then
            echo "Skipping np=$np (matrix size $N not divisible by grid size $grid_size)" >&2
            printf " N/D      |" >> "$SUMMARY_FILE"
            continue
        fi
        
        echo "Running with np=$np processes (grid size $grid_size)..." >&2
        
        # Output file for this run
        OUTPUT_FILE="$RESULTS_DIR/cannon_N${N}_np${np}.out"
        
        # Run Cannon's Algorithm
        result=$(run_with_timeout "mpirun --mca btl_tcp_if_exclude docker0,lo -np $np ./cannon $N")
        
        # Add to summary table
        printf " %s |" "$result" >> "$SUMMARY_FILE"
    done
    
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "All tests completed. Results saved to $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "All tests completed. Results saved to $RESULTS_DIR" >&2
cat "$SUMMARY_FILE" >&2