#!/bin/bash
# Script untuk menjalankan eksperimen topologi MPI
# Membandingkan implementasi standar dengan implementasi topologi Cartesian

# Compile semua program
echo "Compiling programs..."
mpicc -o matvec matvec_fixed.c -lm
mpicc -o matvec_cart matvec_cart.c -lm
mpicc -o cannon_cart cannon_cart.c -lm -lm

# Definisikan file output
OUTPUT_FILE="topology_results_$(date +%s).txt"

# Ukuran matriks yang akan diuji
SIZES=(128 256 512 1024 2048)

# Jumlah proses yang akan diuji
PROCS=(1 2 4 8 16)
# Jika mesin memiliki lebih banyak prosesor, tambahkan
if [ $(nproc) -ge 32 ]; then
    PROCS+=(32)
fi
if [ $(nproc) -ge 64 ]; then
    PROCS+=(64)
fi

# For Cannon's algorithm we need perfect squares
SQUARE_PROCS=(1 4 9 16 25 36 49 64)
# Filter square procs based on available processors
FILTERED_SQUARE_PROCS=()
for np in "${SQUARE_PROCS[@]}"; do
    if [ $np -le $(nproc) ]; then
        FILTERED_SQUARE_PROCS+=($np)
    fi
done

# Tulis header ke file output
echo "Eksperimen Perbandingan Implementasi Standar vs Topologi MPI" | tee -a "$OUTPUT_FILE"
echo "Mesin: $(hostname)" | tee -a "$OUTPUT_FILE"
echo "Jumlah Prosesor: $(nproc)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# === Matrix-Vector Multiplication ===
echo "=== Perkalian Matriks-Vektor: Standar vs Topologi Cartesian ===" | tee -a "$OUTPUT_FILE"
echo "Format: N=ukuran_matriks, NP=jumlah_proses, Waktu=(komputasi/komunikasi) dalam detik" | tee -a "$OUTPUT_FILE"
echo "| N | NP | Standar | Cartesian |" | tee -a "$OUTPUT_FILE"
echo "|---|----|---------|-----------|-" | tee -a "$OUTPUT_FILE"

for n in "${SIZES[@]}"; do
    for np in "${PROCS[@]}"; do
        # Check if matrix size is divisible by process count
        if [ $(($n % $np)) -ne 0 ]; then
            echo "| $n | $np | N/A | N/A |" | tee -a "$OUTPUT_FILE"
            continue
        fi
        
        echo "Running matrix-vector tests with n=$n, np=$np" >&2
        
        # Run standard implementation
        standard_result=$(mpirun -np $np ./matvec $n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$standard_result" ]; then
            standard_result="ERROR"
        fi
        
        # Run Cartesian topology implementation
        cart_result=$(mpirun -np $np ./matvec_cart $n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$cart_result" ]; then
            cart_result="ERROR"
        fi
        
        echo "| $n | $np | $standard_result | $cart_result |" | tee -a "$OUTPUT_FILE"
    done
done

echo "" | tee -a "$OUTPUT_FILE"

# === Cannon's Algorithm ===
echo "=== Perkalian Matriks dengan Algoritma Cannon (Topologi 2D Cartesian) ===" | tee -a "$OUTPUT_FILE"
echo "Format: N=ukuran_matriks, NP=jumlah_proses, Waktu=(komputasi/komunikasi) dalam detik" | tee -a "$OUTPUT_FILE"
echo "| N | NP | Cannon-Cart |" | tee -a "$OUTPUT_FILE"
echo "|---|----|-----------|-" | tee -a "$OUTPUT_FILE"

for n in "${SIZES[@]}"; do
    for np in "${FILTERED_SQUARE_PROCS[@]}"; do
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
        cannon_result=$(mpirun -np $np ./cannon_cart $adjusted_n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$cannon_result" ]; then
            cannon_result="ERROR"
        fi
        
        echo "| $n_display | $np | $cannon_result |" | tee -a "$OUTPUT_FILE"
    done
done

echo "" | tee -a "$OUTPUT_FILE"
echo "Eksperimen selesai. Hasil disimpan ke $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"