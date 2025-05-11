#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <sys/time.h>

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Struct for CSR format
typedef struct {
    int *rowPtr;
    int *colIdx;
    double *values;
    int nnz;
} CSRMatrix;

// Function to convert dense matrix to CSR format
void dense_to_csr(double *A, int N, CSRMatrix *csr) {
    // Count non-zero elements
    int nnz = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i*N + j] != 0.0) {
                nnz++;
            }
        }
    }
    
    // Allocate CSR arrays
    csr->rowPtr = (int*)malloc((N + 1) * sizeof(int));
    csr->colIdx = (int*)malloc(nnz * sizeof(int));
    csr->values = (double*)malloc(nnz * sizeof(double));
    csr->nnz = nnz;
    
    // Fill CSR arrays
    int count = 0;
    csr->rowPtr[0] = 0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i*N + j] != 0.0) {
                csr->colIdx[count] = j;
                csr->values[count] = A[i*N + j];
                count++;
            }
        }
        csr->rowPtr[i + 1] = count;
    }
}

// Function to generate a symmetric positive-definite matrix
void generate_spd_matrix(double *A, double *b, int N) {
    int i, j;
    srand(12345);  // Fixed seed for reproducibility
    
    // First generate a random matrix
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            // Generate random value between -0.5 and 0.5
            A[i*N + j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    
    // Make it symmetric: A = 0.5 * (A + A^T)
    for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {  // Only need to process lower triangle
            double avg = (A[i*N + j] + A[j*N + i]) * 0.5;
            A[i*N + j] = A[j*N + i] = avg;
        }
    }
    
    // Make it diagonally dominant to ensure positive definiteness
    for (i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (j = 0; j < N; j++) {
            if (i != j) {
                row_sum += fabs(A[i*N + j]);
            }
        }
        // Make diagonal elements larger than sum of other elements in row
        A[i*N + i] = row_sum + 1.0;
    }
    
    // Generate right-hand side vector b
    for (i = 0; i < N; i++) {
        b[i] = ((double)rand() / RAND_MAX) * 10.0;
    }
}

// Function to verify the solution using CPU
double verify_solution(double *A, double *x, double *b, int N) {
    double *residual = (double*)malloc(N * sizeof(double));
    
    // Compute residual: r = b - A*x
    for (int i = 0; i < N; i++) {
        residual[i] = b[i];
        for (int j = 0; j < N; j++) {
            residual[i] -= A[i*N + j] * x[j];
        }
    }
    
    // Compute norm ||r||
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        norm += residual[i] * residual[i];
    }
    
    free(residual);
    return sqrt(norm);
}

// Generic CUDA error checking function
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// Custom sparse matrix-vector multiplication kernel
__global__ void spmv_csr_kernel(int num_rows, int* row_ptrs, int* col_indices, 
                              double* values, double* x, double* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        double dot = 0.0;
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_indices[i]];
        }
        
        y[row] = dot;
    }
}

int main(int argc, char *argv[]) {
    // Default parameters
    int N = 1000;        // Matrix size
    int max_iter = 1000;  // Maximum iterations
    double tol = 1e-6;    // Tolerance
    
    // Parse command line arguments
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        max_iter = atoi(argv[2]);
    }
    if (argc > 3) {
        tol = atof(argv[3]);
    }
    
    printf("Conjugate Gradient Method (cuSPARSE)\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Maximum iterations: %d\n", max_iter);
    printf("Tolerance: %e\n", tol);
    
    // Allocate host memory
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    
    if (!A || !b || !x) {
        printf("Host memory allocation failed\n");
        return 1;
    }
    
    // Generate problem
    generate_spd_matrix(A, b, N);
    
    // Initialize solution
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
    
    // Convert to CSR format
    CSRMatrix csr;
    dense_to_csr(A, N, &csr);
    
    printf("Matrix has %d non-zero elements (%.2f%% sparsity)\n", 
           csr.nnz, 100.0 * (1.0 - (double)csr.nnz / (N * N)));
    
    // Allocate device memory
    double *d_b, *d_x, *d_r, *d_p, *d_Ap;
    int *d_csrRowPtr, *d_csrColIdx;
    double *d_csrValues;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_csrRowPtr, (N + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_csrColIdx, csr.nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_csrValues, csr.nnz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Ap, N * sizeof(double)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_csrRowPtr, csr.rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csrColIdx, csr.colIdx, csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csrValues, csr.values, csr.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Setup cuBLAS
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    // Setup kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Timing variables
    double start_time, end_time;
    double compute_time = 0.0;
    double comm_time = 0.0;
    
    // Variables for CG algorithm
    double alpha, beta;
    double r_dot_r, r_dot_r_new, p_dot_Ap;
    int iter;
    
    // Start timer
    start_time = get_time();
    double compute_start, compute_end, comm_start, comm_end;
    
    // Initialize: r0 = b - A*x0, but since x0 = 0, r0 = b
    comm_start = get_time();
    CHECK_CUDA_ERROR(cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice));
    // p0 = r0
    CHECK_CUDA_ERROR(cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));
    comm_end = get_time();
    comm_time += comm_end - comm_start;
    
    // Initial r_dot_r = r0^T * r0
    compute_start = get_time();
    cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r_dot_r);
    compute_end = get_time();
    compute_time += compute_end - compute_start;
    
    double initial_residual = sqrt(r_dot_r);
    printf("Initial residual: %e\n", initial_residual);
    
    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p using our custom SPMV kernel
        compute_start = get_time();
        
        // Use custom kernel for sparse matrix-vector multiplication
        spmv_csr_kernel<<<gridSize, blockSize>>>(N, d_csrRowPtr, d_csrColIdx, d_csrValues, d_p, d_Ap);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Compute p_dot_Ap = p^T * Ap
        cublasDdot(cublasHandle, N, d_p, 1, d_Ap, 1, &p_dot_Ap);
        
        // Compute alpha = r_dot_r / p_dot_Ap
        alpha = r_dot_r / p_dot_Ap;
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Update x = x + alpha*p
        compute_start = get_time();
        cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Update r = r - alpha*Ap
        compute_start = get_time();
        double neg_alpha = -alpha;
        cublasDaxpy(cublasHandle, N, &neg_alpha, d_Ap, 1, d_r, 1);
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Check convergence
        compute_start = get_time();
        cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r_dot_r_new);
        
        if (sqrt(r_dot_r_new) < tol * initial_residual) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Compute beta = r_dot_r_new / r_dot_r
        compute_start = get_time();
        beta = r_dot_r_new / r_dot_r;
        
        // Update p = r + beta*p (first scale p by beta, then add r)
        cublasDscal(cublasHandle, N, &beta, d_p, 1);
        double one = 1.0;
        cublasDaxpy(cublasHandle, N, &one, d_r, 1, d_p, 1);
        
        // Update r_dot_r for next iteration
        r_dot_r = r_dot_r_new;
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Print progress periodically
        if ((iter + 1) % 100 == 0) {
            printf("Iteration %d: residual = %e\n", iter + 1, sqrt(r_dot_r_new));
        }
    }
    
    // Copy result back to host
    comm_start = get_time();
    CHECK_CUDA_ERROR(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    comm_end = get_time();
    comm_time += comm_end - comm_start;
    
    // Stop timer
    end_time = get_time();
    double total_time = end_time - start_time;
    
    // Verify solution
    double residual_norm = verify_solution(A, x, b, N);
    
    // Print results
    printf("\n--- Results ---\n");
    printf("Final residual norm: %e\n", residual_norm);
    printf("Iterations: %d\n", iter);
    printf("Total time: %f seconds\n", total_time);
    printf("Compute time: %f seconds\n", compute_time);
    printf("Communication time: %f seconds\n", comm_time);
    printf("Compute/Comm ratio: %f\n", compute_time / comm_time);
    
    // Clean up
    free(A);
    free(b);
    free(x);
    free(csr.rowPtr);
    free(csr.colIdx);
    free(csr.values);
    
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrValues);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    
    cublasDestroy(cublasHandle);
    
    return 0;
}