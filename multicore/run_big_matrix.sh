
#!/bin/bash
# Script untuk menjalankan eksperimen pada CPU high-end Threadripper (64 prosesor logis)

# Compile program
mpicc -o matvec matvec_fixed.c -lm
mpicc -o matmat matmat_fixed.c -lm
mpicc -o cannon cannon_fixed.c -lm

# Definisikan file output
OUTPUT_FILE="results_big_matrix_$(date +%s).txt"

# Ukuran matriks yang akan diuji (sama seperti sebelumnya)
SIZES=(4096 8192 16384)

# Jumlah proses yang akan diuji
PROCS=(1 2 4 8 16 32 64)

# Tulis header
echo "Eksperimen menggunakan CPU AMD Ryzen Threadripper 3970X (64 prosesor logis)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "=== Eksperimen Perkalian Matriks-Vektor ===" | tee -a "$OUTPUT_FILE"
echo "Format: (waktu_komputasi/waktu_komunikasi) dalam detik" | tee -a "$OUTPUT_FILE"
echo -n "N\t" | tee -a "$OUTPUT_FILE"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

for n in "${SIZES[@]}"; do
    echo -n "$n\t" | tee -a "$OUTPUT_FILE"
    for np in "${PROCS[@]}"; do
        # Jalankan perkalian matriks-vektor
        result=$(mpirun -np $np ./matvec $n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$result" ]; then
            echo -n "ERROR\t" | tee -a "$OUTPUT_FILE"
        else
            echo -n "$result\t" | tee -a "$OUTPUT_FILE"
        fi
    done
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "" | tee -a "$OUTPUT_FILE"
echo "=== Eksperimen Perkalian Matriks-Matriks ===" | tee -a "$OUTPUT_FILE"
echo "Format: (waktu_komputasi/waktu_komunikasi) dalam detik" | tee -a "$OUTPUT_FILE"
echo -n "N\t" | tee -a "$OUTPUT_FILE"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

for n in "${SIZES[@]}"; do
    echo -n "$n\t" | tee -a "$OUTPUT_FILE"
    for np in "${PROCS[@]}"; do
        # Jalankan perkalian matriks-matriks
        result=$(mpirun -np $np ./matmat $n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$result" ]; then
            echo -n "ERROR\t" | tee -a "$OUTPUT_FILE"
        else
            echo -n "$result\t" | tee -a "$OUTPUT_FILE"
        fi
    done
    echo "" | tee -a "$OUTPUT_FILE"
done

# Untuk Cannon, gunakan kuadrat sempurna
SQUARE_PROCS=(1 4 9 16 25 36 49 64)

# Hanya jalankan algoritma Cannon untuk jumlah proses yang merupakan kuadrat sempurna
echo "" | tee -a "$OUTPUT_FILE"
echo "=== Eksperimen Perkalian Matriks dengan Algoritma Cannon ===" | tee -a "$OUTPUT_FILE"
echo "Format: (waktu_komputasi/waktu_komunikasi) dalam detik" | tee -a "$OUTPUT_FILE"
echo -n "N\t" | tee -a "$OUTPUT_FILE"
for np in "${SQUARE_PROCS[@]}"; do
    echo -n "NP=$np\t" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

for n in "${SIZES[@]}"; do
    echo -n "$n\t" | tee -a "$OUTPUT_FILE"
    for np in "${SQUARE_PROCS[@]}"; do
        # Pastikan ukuran matriks dapat dibagi dengan akar kuadrat dari np
        sqrt_np=$(echo "sqrt($np)" | bc)
        adjusted_n=$n
        if [ $(($n % $sqrt_np)) -ne 0 ]; then
            # Sesuaikan ukuran matriks agar dapat dibagi
            adjusted_n=$(( ($n / $sqrt_np) * $sqrt_np ))
            echo -n "($adjusted_n) " | tee -a "$OUTPUT_FILE"
        fi

        # Jalankan algoritma Cannon
        result=$(mpirun -np $np ./cannon $adjusted_n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$result" ]; then
            echo -n "ERROR\t" | tee -a "$OUTPUT_FILE"
        else
            echo -n "$result\t" | tee -a "$OUTPUT_FILE"
        fi
    done
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "" | tee -a "$OUTPUT_FILE"
echo "Hasil telah disimpan ke $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"

