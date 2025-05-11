#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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

// CUDA kernel for matrix-vector multiplication (y = A*x)
__global__ void matrix_vector_multiply(double *A, double *x, double *y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

// CUDA kernel for vector addition (a = b + alpha*c)
__global__ void vector_add(double *a, double *b, double *c, double alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        a[idx] = b[idx] + alpha * c[idx];
    }
}

// CUDA kernel for vector subtraction (a = b - alpha*c)
__global__ void vector_subtract(double *a, double *b, double *c, double alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        a[idx] = b[idx] - alpha * c[idx];
    }
}

// CUDA kernel for vector scaling (y = alpha*x)
__global__ void vector_scale(double *y, double *x, double alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        y[idx] = alpha * x[idx];
    }
}

// CUDA kernel for dot product (result = sum(a[i]*b[i]))
__global__ void dot_product_kernel(double *a, double *b, double *result, int N) {
    __shared__ double cache[256]; // Assuming block size <= 256
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    
    double temp = 0.0;
    while (idx < N) {
        temp += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    
    __syncthreads();
    
    // Perform reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0) {
        result[blockIdx.x] = cache[0];
    }
}

// Host function for computing dot product
double dot_product(double *d_a, double *d_b, int N, cublasHandle_t handle) {
    double result;
    cublasDdot(handle, N, d_a, 1, d_b, 1, &result);
    return result;
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
    
    printf("Conjugate Gradient Method (CUDA)\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Maximum iterations: %d\n", max_iter);
    printf("Tolerance: %e\n", tol);
    
    // Allocate host memory
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    double *x_cublas = (double*)malloc(N * sizeof(double));
    
    if (!A || !b || !x || !x_cublas) {
        printf("Host memory allocation failed\n");
        return 1;
    }
    
    // Generate problem
    generate_spd_matrix(A, b, N);
    
    // Initialize solution
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
        x_cublas[i] = 0.0;
    }
    
    // Allocate device memory
    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;
    
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_b, N * sizeof(double));
    cudaMalloc((void**)&d_x, N * sizeof(double));
    cudaMalloc((void**)&d_r, N * sizeof(double));
    cudaMalloc((void**)&d_p, N * sizeof(double));
    cudaMalloc((void**)&d_Ap, N * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Setup kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
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
    cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice);
    // p0 = r0
    cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice);
    comm_end = get_time();
    comm_time += comm_end - comm_start;
    
    // Initial r_dot_r = r0^T * r0
    compute_start = get_time();
    r_dot_r = dot_product(d_r, d_r, N, handle);
    compute_end = get_time();
    compute_time += compute_end - compute_start;
    
    double initial_residual = sqrt(r_dot_r);
    printf("Initial residual: %e\n", initial_residual);
    
    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p
        compute_start = get_time();
        matrix_vector_multiply<<<gridSize, blockSize>>>(d_A, d_p, d_Ap, N);
        cudaDeviceSynchronize();
        
        // Compute p_dot_Ap = p^T * Ap
        p_dot_Ap = dot_product(d_p, d_Ap, N, handle);
        
        // Compute alpha = r_dot_r / p_dot_Ap
        alpha = r_dot_r / p_dot_Ap;
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Update x = x + alpha*p
        compute_start = get_time();
        vector_add<<<gridSize, blockSize>>>(d_x, d_x, d_p, alpha, N);
        cudaDeviceSynchronize();
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Update r = r - alpha*Ap
        compute_start = get_time();
        vector_subtract<<<gridSize, blockSize>>>(d_r, d_r, d_Ap, alpha, N);
        cudaDeviceSynchronize();
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Check convergence
        compute_start = get_time();
        r_dot_r_new = dot_product(d_r, d_r, N, handle);
        
        if (sqrt(r_dot_r_new) < tol * initial_residual) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
        compute_end = get_time();
        compute_time += compute_end - compute_start;
        
        // Compute beta = r_dot_r_new / r_dot_r
        compute_start = get_time();
        beta = r_dot_r_new / r_dot_r;
        
        // Update p = r + beta*p
        vector_add<<<gridSize, blockSize>>>(d_p, d_r, d_p, beta, N);
        cudaDeviceSynchronize();
        
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
    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
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
    free(x_cublas);
    
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    
    cublasDestroy(handle);
    
    return 0;
}