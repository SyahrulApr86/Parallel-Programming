#!/bin/bash
#SBATCH --job-name=matrix-mult
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --output=results_%j.out
#SBATCH --error=errors_%j.err

# Script tanpa parameter ntasks-per-node, membiarkan SLURM menentukan sesuai ketersediaan

# Compile program
mpicc -o matvec matvec_fixed.c -lm
mpicc -o matmat matmat_fixed.c -lm
mpicc -o cannon cannon_fixed.c -lm

# Ukuran matriks yang akan diuji
SIZES=(128 256 512 1024 2048)

# Jumlah proses yang akan diuji
PROCS=(1 2 4 8)

echo "=== Eksperimen Perkalian Matriks-Vektor ==="
echo "Format: (waktu_komputasi/waktu_komunikasi) dalam detik"
echo -n "N\t"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

for n in "${SIZES[@]}"; do
    echo -n "$n\t"
    for np in "${PROCS[@]}"; do
        # Jalankan perkalian matriks-vektor
        result=$(mpirun -np $np ./matvec $n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$result" ]; then
            echo -n "ERROR\t"
        else
            echo -n "$result\t"
        fi
    done
    echo ""
done

echo ""
echo "=== Eksperimen Perkalian Matriks-Matriks ==="
echo "Format: (waktu_komputasi/waktu_komunikasi) dalam detik"
echo -n "N\t"
for np in "${PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

for n in "${SIZES[@]}"; do
    echo -n "$n\t"
    for np in "${PROCS[@]}"; do
        # Jalankan perkalian matriks-matriks
        result=$(mpirun -np $np ./matmat $n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$result" ]; then
            echo -n "ERROR\t"
        else
            echo -n "$result\t"
        fi
    done
    echo ""
done

# Hanya jalankan algoritma Cannon untuk jumlah proses yang merupakan kuadrat sempurna
echo ""
echo "=== Eksperimen Perkalian Matriks dengan Algoritma Cannon ==="
echo "Format: (waktu_komputasi/waktu_komunikasi) dalam detik"
echo -n "N\t"
# Gunakan jumlah proses yang merupakan kuadrat sempurna
SQUARE_PROCS=(1 4)
for np in "${SQUARE_PROCS[@]}"; do
    echo -n "NP=$np\t"
done
echo ""

for n in "${SIZES[@]}"; do
    echo -n "$n\t"
    for np in "${SQUARE_PROCS[@]}"; do
        # Pastikan ukuran matriks dapat dibagi dengan akar kuadrat dari np
        sqrt_np=$(echo "sqrt($np)" | bc)
        adjusted_n=$n
        if [ $(($n % $sqrt_np)) -ne 0 ]; then
            # Sesuaikan ukuran matriks agar dapat dibagi
            adjusted_n=$(( ($n / $sqrt_np) * $sqrt_np ))
            echo -n "($adjusted_n) "
        fi
        
        # Jalankan algoritma Cannon
        result=$(mpirun -np $np ./cannon $adjusted_n 2>/dev/null | grep "Result summary" | awk '{print $3}')
        if [ -z "$result" ]; then
            echo -n "ERROR\t"
        else
            echo -n "$result\t"
        fi
    done
    echo ""
done
