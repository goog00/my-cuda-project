#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h> // 添加这个头文件


// Helper macro for CUDA error checking
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void gemm_coalesced_only(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
	
	int row = blockIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	if (row < M && col < N) {
	    float sum = 0.0f;

	    for (int k = 0; k < K; ++k){
	        sum += A[row * K + k] * B[k * N + col];
	    }

	    int idx = row * N + col;
	    C[idx] = alpha * sum + beta * C[idx];
	}
}

extern "C" void solve_coalesced_only(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
	const int BLOCK_X = 256;
	dim3 block(BLOCK_X);
	dim3 grid((N + BLOCK_X -1) / BLOCK_X, M); // 修复语法错误

	gemm_coalesced_only<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);

	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaDeviceSynchronize());

}



int main() {
	const int M = 512, N =512, K = 512;
	const float alpha = 1.0f, beta = 0.0f;

	size_t sizeA = M * K * sizeof(float);
	size_t sizeB = K * N * sizeof(float);
	size_t sizeC = M * N * sizeof(float);

	// allocate host memory
	float * A_h = (float*)malloc(sizeA);
	float * B_h = (float*)malloc(sizeB);
	float * C_h = (float*)malloc(sizeC);

	//alocate device memory
	float * A_d, *B_d, *C_d;
	cudaMalloc(&A_d, sizeA);
	cudaMalloc(&B_d, sizeB);
	cudaMalloc(&C_d, sizeC);

	//initialize host data
	for (int i = 0; i < M * K; ++i) A_h[i] = 1.0f;
	for (int i = 0; i < K * N; ++i) B_h[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) C_h[i] = 0.0f;



    //copy data to device
    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice);  // 修复这里的字符错误
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, sizeC, cudaMemcpyHostToDevice);

	//run gemm multiple times to amplify kernel time for profiling
	const int REPS = 1000;
	for (int i = 0; i < REPS; ++i) {
		solve_coalesced_only(A_d, B_d, C_d, M, N, K, alpha, beta);
	}

    //copy result back to host
    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);

    //print a sample result
    std::cout << "Sample result - C[0][0] = " << C_h[0] << std::endl;
    std::cout << "Expected value for C[0][0] = " << K << " (if alpha=1, beta=0)" << std::endl;


    //clean up
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;

}

















