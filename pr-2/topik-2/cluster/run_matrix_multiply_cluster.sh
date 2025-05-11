#!/bin/bash
#SBATCH --job-name=matmul-cluster
#SBATCH --partition=batch
#SBATCH --nodes=1-8        # Request up to 8 nodes
#SBATCH --output=matmul_cluster_%j.out
#SBATCH --error=matmul_cluster_%j.err
#SBATCH --time=72:00:00    # Set maximum run time to 72 hours

# Compile the program
mpicc -o matrix_multiply_mpi matrix_multiply_mpi.c -lm -O3

# Matrix sizes to test
SIZES=(128 256 512 1024 2048)

# Process counts to test
PROCS=(1 2 4 8)

# Create results directory
RESULTS_DIR="matrix_multiply_results_cluster"
mkdir -p "$RESULTS_DIR"

# Create CSV file for results
RESULTS_CSV="$RESULTS_DIR/matrix_multiply_results_cluster.csv"
echo "N,np,ComputeTime,CommunicationTime,TotalTime" > "$RESULTS_CSV"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Matrix Multiplication - Cluster Environment Results" > "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Environment B (8 Nodes Cluster)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Format: ComputeTime/CommunicationTime (seconds)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Calculate total available processes
TOTAL_PROCS=$(($SLURM_JOB_NUM_NODES * 8))
echo "Running on $SLURM_JOB_NUM_NODES nodes with approximately $TOTAL_PROCS total processes"
echo "" | tee -a "$SUMMARY_FILE"

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

# Function to run a command with timeout
run_with_timeout() {
    local cmd="$1"
    local timeout=3600  # 1 hour timeout for each test case
    
    # Create a temporary file for output
    local outfile=$(mktemp)
    
    # Print the command we're about to run (to stderr only)
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
        local compute_time=$(grep "Compute time:" "$outfile" | awk '{print $3}')
        local comm_time=$(grep "Communication time:" "$outfile" | awk '{print $3}')
        local total_time=$(grep "Total time:" "$outfile" | awk '{print $3}')
        
        echo "$compute_time/$comm_time"  # Return format for summary table
        
        # Add to CSV
        echo "$N,$np,$compute_time,$comm_time,$total_time" >> "$RESULTS_CSV"
    fi
    
    # Copy the output to the result file
    cp "$outfile" "$OUTPUT_FILE"
    
    # Clean up
    rm -f "$outfile"
}

# Run tests for each matrix size and process count
for N in "${SIZES[@]}"; do
    echo "====================================================="
    echo "Running Matrix Multiplication on cluster with N=$N"
    echo "====================================================="
    
    # Start row in summary table
    printf "N=%-4s|" "$N" >> "$SUMMARY_FILE"
    
    for np in "${PROCS[@]}"; do
        # Skip if np is more than available processes
        if [ $np -gt $TOTAL_PROCS ]; then
            printf " N/A       |" >> "$SUMMARY_FILE"
            continue
        fi
        
        echo "Running with np=$np processes..." >&2
        
        # Output file for this run
        OUTPUT_FILE="$RESULTS_DIR/matmul_N${N}_np${np}.out"
        
        # Run the matrix multiplication
        result=$(run_with_timeout "mpirun --mca btl_tcp_if_exclude docker0,lo -np $np ./matrix_multiply_mpi $N")
        
        # Add to summary table
        printf " %s |" "$result" >> "$SUMMARY_FILE"
    done
    
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "All tests completed. Results saved to $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "All tests completed. Results saved to $RESULTS_DIR" >&2
cat "$SUMMARY_FILE" >&2
