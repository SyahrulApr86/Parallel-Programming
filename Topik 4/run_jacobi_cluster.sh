#!/bin/bash
#SBATCH --job-name=jacobi-cluster
#SBATCH --partition=batch
#SBATCH --nodes=1-8        # Request up to 8 nodes
#SBATCH --output=jacobi_cluster_%j.out
#SBATCH --error=jacobi_cluster_%j.err

# Compile the program
mpicc -o jacobi_solver jacobi_solver.c -lm

# Matrix sizes to test
SIZES=(128 256 512 1024 2048 4056 8112 16224)

# Process counts to test
PROCS=(1 2 4 8 16 32)

# Create results directory
RESULTS_DIR="jacobi_results_cluster"
mkdir -p "$RESULTS_DIR"

# Create CSV file for results
RESULTS_CSV="$RESULTS_DIR/jacobi_results_cluster.csv"
echo "N,np,Iterations,FinalResidual,TotalTime,ComputeTime,CommunicationTime,FinalError" > "$RESULTS_CSV"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Jacobi Iteration - Cluster Environment Results" > "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Environment A (8 Nodes Cluster)" >> "$SUMMARY_FILE"
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

# Run tests for each matrix size and process count
for N in "${SIZES[@]}"; do
    echo "====================================================="
    echo "Running Jacobi solver on cluster with N=$N"
    echo "====================================================="
    
    # Start row in summary table
    printf "N=%-4s|" "$N" >> "$SUMMARY_FILE"
    
    for np in "${PROCS[@]}"; do
        echo "Running with np=$np processes..."
        
        # Output file for this run
        OUTPUT_FILE="$RESULTS_DIR/jacobi_N${N}_np${np}.out"
        
        # Run the Jacobi solver
        mpirun --mca btl_tcp_if_exclude docker0,lo -np $np ./jacobi_solver $N > "$OUTPUT_FILE"
        
        # Extract all information
        ITERATIONS=$(grep "Iterations:" "$OUTPUT_FILE" | awk '{print $2}')
        FINAL_RESIDUAL=$(grep "Final residual:" "$OUTPUT_FILE" | awk '{print $3}')
        TOTAL_TIME=$(grep "Total time:" "$OUTPUT_FILE" | awk '{print $3}')
        COMPUTE_TIME=$(grep "Compute time:" "$OUTPUT_FILE" | awk '{print $3}')
        COMM_TIME=$(grep "Communication time:" "$OUTPUT_FILE" | awk '{print $3}')
        FINAL_ERROR=$(grep "Final error" "$OUTPUT_FILE" | awk '{print $4}')
        
        # Save full output for reference
        cp "$OUTPUT_FILE" "$RESULTS_DIR/full_output_N${N}_np${np}.txt"
        
        # Add to CSV
        echo "$N,$np,$ITERATIONS,$FINAL_RESIDUAL,$TOTAL_TIME,$COMPUTE_TIME,$COMM_TIME,$FINAL_ERROR" >> "$RESULTS_CSV"
        
        # Add to summary table
        printf " %s/%s |" "$COMPUTE_TIME" "$COMM_TIME" >> "$SUMMARY_FILE"
    done
    
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "All tests completed. Results saved to $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "All tests completed. Results saved to $RESULTS_DIR"
cat "$SUMMARY_FILE"