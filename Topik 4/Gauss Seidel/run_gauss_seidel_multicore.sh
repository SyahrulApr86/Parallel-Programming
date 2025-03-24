#!/bin/bash
# Script to run Gauss-Seidel iteration on a multicore system (1 CPU with 64 cores)

# Compile the program
mpicc -o gauss_seidel_solver gauss_seidel_solver.c -lm

# Matrix sizes to test
SIZES=(128 256 512 1024 2048 4056 8112 16224)

# Process counts to test
PROCS=(1 2 4 8 16 32 64)

# Create results directory
RESULTS_DIR="gauss_seidel_results_multicore"
mkdir -p "$RESULTS_DIR"

# Create CSV file for results
RESULTS_CSV="$RESULTS_DIR/gauss_seidel_results_multicore.csv"
echo "N,np,ComputeTime,CommunicationTime,TotalTime,Iterations" > "$RESULTS_CSV"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Gauss-Seidel Iteration - Multicore Environment Results" > "$SUMMARY_FILE"
echo "=================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Environment B (Multicore System - 1 CPU with 64 cores)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Format: ComputeTime/CommunicationTime (seconds)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

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
    local timeout=3600  # 1 hour timeout for larger matrices
    
    # Create a temporary file for output
    local outfile=$(mktemp)
    
    # Run the command with timeout
    timeout --kill-after=30s $timeout bash -c "$cmd" > "$outfile" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "TIMEOUT"
        echo "Command timed out after ${timeout}s: $cmd" >&2
        echo "TIMEOUT/TIMEOUT"  # Return format for summary table
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR($exit_code)"
        echo "Command failed with exit code $exit_code: $cmd" >&2
        echo "ERROR/ERROR"  # Return format for summary table
    else
        # Extract results from the output file
        local compute_time=$(grep "Compute time:" "$outfile" | awk '{print $3}')
        local comm_time=$(grep "Communication time:" "$outfile" | awk '{print $3}')
        local total_time=$(grep "Total time:" "$outfile" | awk '{print $3}')
        local iterations=$(grep "Iterations:" "$outfile" | awk '{print $2}')
        
        echo "$compute_time/$comm_time"  # Return format for summary table
        
        # Add to CSV
        echo "$N,$np,$compute_time,$comm_time,$total_time,$iterations" >> "$RESULTS_CSV"
    fi
    
    # Copy the output to the result file
    cp "$outfile" "$OUTPUT_FILE"
    
    # Clean up
    rm -f "$outfile"
}

# Run tests for each matrix size and process count
for N in "${SIZES[@]}"; do
    echo "====================================================="
    echo "Running Gauss-Seidel solver on multicore with N=$N"
    echo "====================================================="
    
    # Start row in summary table
    printf "N=%-4s|" "$N" >> "$SUMMARY_FILE"
    
    for np in "${PROCS[@]}"; do
        echo "Running with np=$np processes..."
        
        # Output file for this run
        OUTPUT_FILE="$RESULTS_DIR/gs_N${N}_np${np}.out"
        
        # Run the Gauss-Seidel solver
        result=$(run_with_timeout "mpirun -np $np ./gauss_seidel_solver $N")
        
        # Add to summary table
        printf " %s |" "$result" >> "$SUMMARY_FILE"
    done
    
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "All tests completed. Results saved to $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "All tests completed. Results saved to $RESULTS_DIR"
cat "$SUMMARY_FILE"