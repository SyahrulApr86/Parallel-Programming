#!/bin/bash
# Script untuk menjalankan simulasi CFD pada mesin multicore

# Kompilasi program
echo "Compiling CFD simulation code..."
mpicc -o cfd_simulation cfd_simulation.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Definisikan file output
OUTPUT_FILE="cfd_results_$(date +%s).txt"

# Domain sizes to test (width x height)
DOMAIN_SIZES=("100 50" "200 100" "400 100" "800 200")

# Reynolds numbers to test
REYNOLDS=(10 100 500 1000)

# Process counts to test (perfect squares for 2D grid)
PROCS=(4 9 16 25 36)

# Tulis header
echo "==== CFD Simulation Experiments using Process Topologies ====" | tee "$OUTPUT_FILE"
echo "Execution timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Eksperimen dengan jumlah proses yang berbeda
echo "=== CFD Simulation with Different Process Counts ===" | tee -a "$OUTPUT_FILE"
echo "Domain: 400x100, Reynolds: 100" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "NP\t" | tee -a "$OUTPUT_FILE"
for np in "${PROCS[@]}"; do
    echo -n "$np\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"
for np in "${PROCS[@]}"; do
    echo "Running CFD simulation with $np processes..." >&2
    
    # Jalankan CFD dengan timeout 10 menit
    result=$(timeout 600s mpirun -np $np ./cfd_simulation 400 100 100 2>/dev/null)
    
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

# Eksperimen dengan ukuran domain yang berbeda
echo "" | tee -a "$OUTPUT_FILE"
echo "=== CFD Simulation with Different Domain Sizes ===" | tee -a "$OUTPUT_FILE"
echo "Process count: 16, Reynolds: 100" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Domain\t" | tee -a "$OUTPUT_FILE"
for domain in "${DOMAIN_SIZES[@]}"; do
    echo -n "$domain\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"
for domain in "${DOMAIN_SIZES[@]}"; do
    nx=$(echo $domain | cut -d' ' -f1)
    ny=$(echo $domain | cut -d' ' -f2)
    
    echo "Running CFD simulation with domain $domain..." >&2
    
    # Jalankan CFD dengan timeout 10 menit
    result=$(timeout 600s mpirun -np 16 ./cfd_simulation $nx $ny 100 2>/dev/null)
    
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

# Eksperimen dengan bilangan Reynolds yang berbeda
echo "" | tee -a "$OUTPUT_FILE"
echo "=== CFD Simulation with Different Reynolds Numbers ===" | tee -a "$OUTPUT_FILE"
echo "Domain: 400x100, Process count: 16" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Re\t" | tee -a "$OUTPUT_FILE"
for re in "${REYNOLDS[@]}"; do
    echo -n "$re\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"
for re in "${REYNOLDS[@]}"; do
    echo "Running CFD simulation with Reynolds number $re..." >&2
    
    # Jalankan CFD dengan timeout 10 menit
    result=$(timeout 600s mpirun -np 16 ./cfd_simulation 400 100 $re 2>/dev/null)
    
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
echo "=== CFD Simulation with CPU Binding ===" | tee -a "$OUTPUT_FILE"
echo "Domain: 400x100, Reynolds: 100, Process count: 16" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Binding\t" | tee -a "$OUTPUT_FILE"
echo -n "None\t" | tee -a "$OUTPUT_FILE"
echo -n "Core\t" | tee -a "$OUTPUT_FILE"
echo -n "Socket" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"

# Without binding
echo "Running CFD simulation without binding..." >&2
result=$(timeout 600s mpirun -np 16 ./cfd_simulation 400 100 100 2>/dev/null)
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
echo "Running CFD simulation with core binding..." >&2
result=$(timeout 600s mpirun -np 16 --bind-to core ./cfd_simulation 400 100 100 2>/dev/null)
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
echo "Running CFD simulation with socket binding..." >&2
result=$(timeout 600s mpirun -np 16 --bind-to socket ./cfd_simulation 400 100 100 2>/dev/null)
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
echo "=== CFD Simulation with/without Reordering ===" | tee -a "$OUTPUT_FILE"
echo "Domain: 400x100, Reynolds: 100, Process count: 16" | tee -a "$OUTPUT_FILE"
echo "Format: (computation_time/communication_time) in seconds" | tee -a "$OUTPUT_FILE"
echo -n "Reordering\t" | tee -a "$OUTPUT_FILE"
echo -n "With\t" | tee -a "$OUTPUT_FILE"
echo -n "Without" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo -n "Time\t" | tee -a "$OUTPUT_FILE"

# With reordering (default)
echo "Running CFD simulation with reordering..." >&2
result=$(timeout 600s mpirun -np 16 ./cfd_simulation 400 100 100 2>/dev/null)
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
cp cfd_simulation.c cfd_simulation_no_reorder.c
sed -i 's/reorder = 1;/reorder = 0;/' cfd_simulation_no_reorder.c
mpicc -o cfd_simulation_no_reorder cfd_simulation_no_reorder.c -lm

# Without reordering
echo "Running CFD simulation without reordering..." >&2
result=$(timeout 600s mpirun -np 16 ./cfd_simulation_no_reorder 400 100 100 2>/dev/null)
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