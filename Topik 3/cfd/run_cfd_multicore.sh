#!/bin/bash
# Script to run CFD simulation on a multicore system (1 CPU with 64 cores)

# Compile the program
mpicc -o cfd_simulation cfd_simulation.c -lm -O3

# Domain sizes to test (NX x NY)
DOMAIN_SIZES=(
    "100 50"     # Small domain
    "200 100"    # Medium domain
    "400 200"    # Large domain
    "800 400"    # Extra large domain
)

# Reynolds numbers to test
REYNOLDS=(100 500 1000)

# Process counts to test
PROCS=(1 2 4 8 16 32 64)

# Create results directory
RESULTS_DIR="cfd_results_multicore"
mkdir -p "$RESULTS_DIR"

# Create CSV file for results
RESULTS_CSV="$RESULTS_DIR/cfd_results_multicore.csv"
echo "NX,NY,Reynolds,np,ComputeTime,CommunicationTime,TotalTime" > "$RESULTS_CSV"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "CFD Simulation - Multicore Environment Results" > "$SUMMARY_FILE"
echo "============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Environment B (Multicore System - 1 CPU with 64 cores)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Format: ComputeTime/CommunicationTime (seconds)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to run a command with timeout
run_with_timeout() {
    local cmd="$1"
    local timeout=7200  # 2 hour timeout for each test case
    
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
        echo "$NX,$NY,$RE,$np,$compute_time,$comm_time,$total_time" >> "$RESULTS_CSV"
    fi
    
    # Copy the output to the result file
    cp "$outfile" "$OUTPUT_FILE"
    
    # Clean up
    rm -f "$outfile"
}

# Run tests for different domain sizes, Reynolds numbers and process counts
for DOMAIN in "${DOMAIN_SIZES[@]}"; do
    # Split domain string into NX and NY
    read NX NY <<< "$DOMAIN"
    
    for RE in "${REYNOLDS[@]}"; do
        # Create table header for this configuration
        echo "" >> "$SUMMARY_FILE"
        echo "Domain Size: ${NX}x${NY}, Reynolds: $RE" >> "$SUMMARY_FILE"
        echo -n "Processes |" >> "$SUMMARY_FILE"
        for np in "${PROCS[@]}"; do
            printf " np=%-4s |" "$np" >> "$SUMMARY_FILE"
        done
        echo "" >> "$SUMMARY_FILE"
        
        echo -n "----------|" >> "$SUMMARY_FILE"
        for np in "${PROCS[@]}"; do
            echo -n "----------|" >> "$SUMMARY_FILE"
        done
        echo "" >> "$SUMMARY_FILE"
        
        echo "============================================================"
        echo "Running CFD simulation with domain ${NX}x${NY}, Reynolds $RE"
        echo "============================================================"
        
        # Start row in summary table
        printf "%-10s|" "Comp/Comm" >> "$SUMMARY_FILE"
        
        for np in "${PROCS[@]}"; do
            echo "Running with np=$np processes..." >&2
            
            # Output file for this run
            OUTPUT_FILE="$RESULTS_DIR/cfd_${NX}x${NY}_Re${RE}_np${np}.out"
            
            # Run the CFD simulation
            result=$(run_with_timeout "mpirun -np $np ./cfd_simulation $NX $NY $RE")
            
            # Add to summary table
            printf " %s |" "$result" >> "$SUMMARY_FILE"
        done
        
        echo "" >> "$SUMMARY_FILE"
    done
done

echo "" >> "$SUMMARY_FILE"
echo "All tests completed. Results saved to $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "All tests completed. Results saved to $RESULTS_DIR" >&2
cat "$SUMMARY_FILE" >&2