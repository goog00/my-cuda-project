#include <cuda_runtime.h>
#include <iostream>

// Native GEMM kernel: C = alpha * A * B + beta * C
// Simple implementation where each thread computes one element of C
__global__ void gemm_native(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K,
                           float alpha, float beta) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Apply alpha and beta coefficients
        int idx = row * N + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

// Host wrapper function to launch the kernel
extern "C" void solve_native(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta) {
    // Define block and grid dimensions
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
              
    // Launch the kernel
    gemm_native<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

int main() {
    // Matrix dimensions
    const int M = 512, N = 512, K = 512;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Calculate memory sizes
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory
    float *A_h = (float*)malloc(sizeA);
    float *B_h = (float*)malloc(sizeB);
    float *C_h = (float*)malloc(sizeC);
    
    // Allocate device memory
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeA);
    cudaMalloc(&B_d, sizeB);
    cudaMalloc(&C_d, sizeC);
    
    // Initialize host data
    for (int i = 0; i < M * K; ++i) A_h[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B_h[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) C_h[i] = 0.0f;
    
    // Copy data to device
    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, sizeC, cudaMemcpyHostToDevice);
    
    // Run the GEMM computation
    solve_native(A_d, B_d, C_d, M, N, K, alpha, beta);
    
    // Copy result back to host
    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);
    
    // Print a sample result
    std::cout << "Sample result - C[0][0] = " << C_h[0] << std::endl;
    std::cout << "Expected value for C[0][0] = " << K << " (if alpha=1, beta=0)" << std::endl;
    
    // Cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
    
    return 0;
}