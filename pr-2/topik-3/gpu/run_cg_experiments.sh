#!/bin/bash
# Script to run Conjugate Gradient experiments on GPU

# Compile all versions
echo "Compiling CUDA implementations..."
nvcc -o cg_cuda conjugate_gradient_cuda.cu -lcudart -lcublas -O3
nvcc -o cg_cublas conjugate_gradient_cublas.cu -lcudart -lcublas -O3
nvcc -o cg_cusparse conjugate_gradient_cusparse.cu -lcudart -lcublas -lcusparse -O3

# Matrix sizes to test
SIZES=(1000 2000 5000 10000 20000)

# Create results directory
RESULTS_DIR="cg_gpu_results"
mkdir -p "$RESULTS_DIR"

# Create CSV file for results
RESULTS_CSV="$RESULTS_DIR/cg_gpu_results.csv"
echo "Implementation,N,Iterations,ComputeTime,CommunicationTime,TotalTime,Residual" > "$RESULTS_CSV"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Conjugate Gradient Method - GPU Implementations" > "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Environment: NVIDIA GTX 1080 Ti GPU" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Format: ComputeTime/CommunicationTime (seconds)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Table header
echo -n "       |" >> "$SUMMARY_FILE"
for size in "${SIZES[@]}"; do
    printf " N=%-6s |" "$size" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"

echo -n "-------|" >> "$SUMMARY_FILE"
for size in "${SIZES[@]}"; do
    echo -n "-----------|" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"

# Function to run an experiment and extract results
run_experiment() {
    local impl=$1
    local executable=$2
    local size=$3
    local max_iter=5000
    local tol=1e-6
    
    echo "Running $impl with N=$size..."
    
    # Run the implementation and capture output
    local output_file="$RESULTS_DIR/${impl}_N${size}.out"
    $executable $size $max_iter $tol > "$output_file" 2>&1
    
    # Extract results
    local iterations=$(grep "Converged after" "$output_file" | awk '{print $3}' || echo $max_iter)
    local compute_time=$(grep "Compute time:" "$output_file" | awk '{print $3}' || echo "N/A")
    local comm_time=$(grep "Communication time:" "$output_file" | awk '{print $3}' || echo "N/A")
    local total_time=$(grep "Total time:" "$output_file" | awk '{print $3}' || echo "N/A")
    local residual=$(grep "Final residual norm:" "$output_file" | awk '{print $4}' || echo "N/A")
    
    # Add to CSV
    echo "$impl,$size,$iterations,$compute_time,$comm_time,$total_time,$residual" >> "$RESULTS_CSV"
    
    # Return formatted output for summary table
    if [[ "$compute_time" != "N/A" && "$comm_time" != "N/A" ]]; then
        echo "$compute_time/$comm_time"
    else
        echo "FAILED"
    fi
}

# Run experiments for each implementation
for impl in "CUDA" "cuBLAS" "cuSPARSE"; do
    echo "Testing $impl implementation..."
    echo -n "$impl |" >> "$SUMMARY_FILE"
    
    case $impl in
        "CUDA")
            executable="./cg_cuda"
            ;;
        "cuBLAS")
            executable="./cg_cublas"
            ;;
        "cuSPARSE")
            executable="./cg_cusparse"
            ;;
    esac
    
    for size in "${SIZES[@]}"; do
        result=$(run_experiment $impl $executable $size)
        printf " %s |" "$result" >> "$SUMMARY_FILE"
    done
    
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "All tests completed. Results saved to $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "All tests completed. Results saved to $RESULTS_DIR"
cat "$SUMMARY_FILE"