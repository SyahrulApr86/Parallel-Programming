#!/bin/bash
# Script untuk menjalankan simulasi penyebaran panas 2D dengan MPI Process Topologies

# Kompilasi program
echo "Compiling heat diffusion simulation code..."
mpicc -o heat_diffusion heat_diffusion.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Definisikan file output
OUTPUT_FILE="heat_results_$(date +%s).txt"

# Grid sizes to test
GRID_SIZES=(120 240 480 960)

# Iteration counts to test
ITERATIONS=(100 500 1000)

# Process counts to test (perfect squares for 2D grid)
PROCS=(4 9 16 25)

# Tulis header
echo "==== Heat Diffusion 2D Experiments with MPI Process Topologies ====" | tee "$OUTPUT_FILE"
echo "Execution timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Eksperimen dengan jumlah proses yang berbeda
echo "=== Heat Diffusion with Different Process Counts ===" | tee -a "$OUTPUT_FILE"
echo "Grid size: 240x240, Iterations: 100" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "NP\t" | tee -a "$OUTPUT_FILE"
for np in "${PROCS[@]}"; do
    echo -n "$np\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"
for np in "${PROCS[@]}"; do
    echo "Running heat diffusion with $np processes..." >&2
    
    # Run with timeout
    result=$(timeout 300s mpirun -np $np ./heat_diffusion 240 100 2>/dev/null)
    
    if [ $? -eq 124 ]; then
        echo -n "TIMEOUT\t" | tee -a "$OUTPUT_FILE"
    else
        # Extract result summary
        summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
        if [ -z "$summary" ]; then
            echo -n "FAILED\t" | tee -a "$OUTPUT_FILE"
        else
            echo -n "$summary\t" | tee -a "$OUTPUT_FILE"
        fi
    fi
done
echo "" | tee -a "$OUTPUT_FILE"

# Eksperimen dengan ukuran grid yang berbeda
echo "" | tee -a "$OUTPUT_FILE"
echo "=== Heat Diffusion with Different Grid Sizes ===" | tee -a "$OUTPUT_FILE"
echo "Process count: 16, Iterations: 100" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Size\t" | tee -a "$OUTPUT_FILE"
for size in "${GRID_SIZES[@]}"; do
    echo -n "${size}x${size}\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"
for size in "${GRID_SIZES[@]}"; do
    echo "Running heat diffusion with grid size ${size}x${size}..." >&2
    
    # Run with timeout
    result=$(timeout 300s mpirun -np 16 ./heat_diffusion $size 100 2>/dev/null)
    
    if [ $? -eq 124 ]; then
        echo -n "TIMEOUT\t" | tee -a "$OUTPUT_FILE"
    else
        # Extract result summary
        summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
        if [ -z "$summary" ]; then
            echo -n "FAILED\t" | tee -a "$OUTPUT_FILE"
        else
            echo -n "$summary\t" | tee -a "$OUTPUT_FILE"
        fi
    fi
done
echo "" | tee -a "$OUTPUT_FILE"

# Eksperimen dengan iterasi berbeda
echo "" | tee -a "$OUTPUT_FILE"
echo "=== Heat Diffusion with Different Iteration Counts ===" | tee -a "$OUTPUT_FILE"
echo "Grid size: 240x240, Process count: 16" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Iter\t" | tee -a "$OUTPUT_FILE"
for iter in "${ITERATIONS[@]}"; do
    echo -n "$iter\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"
for iter in "${ITERATIONS[@]}"; do
    echo "Running heat diffusion with $iter iterations..." >&2
    
    # Run with timeout
    result=$(timeout 300s mpirun -np 16 ./heat_diffusion 240 $iter 2>/dev/null)
    
    if [ $? -eq 124 ]; then
        echo -n "TIMEOUT\t" | tee -a "$OUTPUT_FILE"
    else
        # Extract result summary
        summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
        if [ -z "$summary" ]; then
            echo -n "FAILED\t" | tee -a "$OUTPUT_FILE"
        else
            echo -n "$summary\t" | tee -a "$OUTPUT_FILE"
        fi
    fi
done
echo "" | tee -a "$OUTPUT_FILE"

# Eksperimen dengan/tanpa binding ke core
echo "" | tee -a "$OUTPUT_FILE"
echo "=== Heat Diffusion with CPU Binding ===" | tee -a "$OUTPUT_FILE"
echo "Grid size: 240x240, Process count: 16, Iterations: 100" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Binding\t" | tee -a "$OUTPUT_FILE"
echo -n "None\t" | tee -a "$OUTPUT_FILE"
echo -n "Core\t" | tee -a "$OUTPUT_FILE"
echo -n "Socket" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"

# Without binding
echo "Running heat diffusion without binding..." >&2
result=$(timeout 300s mpirun -np 16 ./heat_diffusion 240 100 2>/dev/null)
if [ $? -eq 124 ]; then
    echo -n "TIMEOUT\t" | tee -a "$OUTPUT_FILE"
else
    summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
    if [ -z "$summary" ]; then
        echo -n "FAILED\t" | tee -a "$OUTPUT_FILE"
    else
        echo -n "$summary\t" | tee -a "$OUTPUT_FILE"
    fi
fi

# With core binding
echo "Running heat diffusion with core binding..." >&2
result=$(timeout 300s mpirun -np 16 --bind-to core ./heat_diffusion 240 100 2>/dev/null)
if [ $? -eq 124 ]; then
    echo -n "TIMEOUT\t" | tee -a "$OUTPUT_FILE"
else
    summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
    if [ -z "$summary" ]; then
        echo -n "FAILED\t" | tee -a "$OUTPUT_FILE"
    else
        echo -n "$summary\t" | tee -a "$OUTPUT_FILE"
    fi
fi

# With socket binding
echo "Running heat diffusion with socket binding..." >&2
result=$(timeout 300s mpirun -np 16 --bind-to socket ./heat_diffusion 240 100 2>/dev/null)
if [ $? -eq 124 ]; then
    echo -n "TIMEOUT" | tee -a "$OUTPUT_FILE"
else
    summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
    if [ -z "$summary" ]; then
        echo -n "FAILED" | tee -a "$OUTPUT_FILE"
    else
        echo -n "$summary" | tee -a "$OUTPUT_FILE"
    fi
fi
echo "" | tee -a "$OUTPUT_FILE"

# Eksperimen dengan/tanpa reordering
echo "" | tee -a "$OUTPUT_FILE"
echo "=== Heat Diffusion with/without Reordering ===" | tee -a "$OUTPUT_FILE"
echo "Grid size: 240x240, Process count: 16, Iterations: 100" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Reordering\t" | tee -a "$OUTPUT_FILE"
echo -n "With\t" | tee -a "$OUTPUT_FILE"
echo -n "Without" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"

# With reordering (default)
echo "Running heat diffusion with reordering..." >&2
result=$(timeout 300s mpirun -np 16 ./heat_diffusion 240 100 2>/dev/null)
if [ $? -eq 124 ]; then
    echo -n "TIMEOUT\t" | tee -a "$OUTPUT_FILE"
else
    summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
    if [ -z "$summary" ]; then
        echo -n "FAILED\t" | tee -a "$OUTPUT_FILE"
    else
        echo -n "$summary\t" | tee -a "$OUTPUT_FILE"
    fi
fi

# Create a version without reordering
cp heat_diffusion.c heat_diffusion_no_reorder.c
sed -i 's/reorder = 1;/reorder = 0;/' heat_diffusion_no_reorder.c
mpicc -o heat_diffusion_no_reorder heat_diffusion_no_reorder.c -lm

# Without reordering
echo "Running heat diffusion without reordering..." >&2
result=$(timeout 300s mpirun -np 16 ./heat_diffusion_no_reorder 240 100 2>/dev/null)
if [ $? -eq 124 ]; then
    echo -n "TIMEOUT" | tee -a "$OUTPUT_FILE"
else
    summary=$(echo "$result" | grep "Result summary" | awk '{print $3}')
    if [ -z "$summary" ]; then
        echo -n "FAILED" | tee -a "$OUTPUT_FILE"
    else
        echo -n "$summary" | tee -a "$OUTPUT_FILE"
    fi
fi
echo "" | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "All experiments completed. Results are saved in $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"