#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void gemm_tiled_smem(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C,
                               int M, int N, int K,
                               float alpha, float beta) {
    // block 负责的 C tile 左上角 (blockRow, blockCol)
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // 线程在 block 内的 2D 坐标
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    // 每线程负责的 C 子块起点（8x8）
    int row0 = blockRow + ty * TM;  // 每个 ty 跨 8 行
    int col0 = blockCol + tx * TN;  // 每个 tx 跨 8 列

    __shared__ half As[BM][BK]; // 128x16 = 2048 half = 4KB
    __shared__ half Bs[BK][BN]; // 16x128 = 2048 half = 4KB

    // FP32 accumulators：8x8
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // K 分块
    for (int k0 = 0; k0 < K; k0 += BK) {
        // -----------------------------
        // 1) 协作加载 A tile: As[0..127][0..15]
        // 每线程加载 8 个 half：覆盖 2048 元素/256线程=8
        // -----------------------------
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            int r = blockRow + ty * TM + i; // 8 行
            int c = k0 + tx;                // BK=16 列由 tx 覆盖
            half v = __float2half(0.0f);
            if (r < M && c < K) v = A[r * K + c];
            As[ty * TM + i][tx] = v;
        }

        // -----------------------------
        // 2) 协作加载 B tile: Bs[0..15][0..127]
        // 同样每线程加载 8 个 half：覆盖 2048 元素
        // -----------------------------
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = k0 + ty;                // BK=16 行由 ty 覆盖
            int c = blockCol + tx * TN + j; // 8 列
            half v = __float2half(0.0f);
            if (r < K && c < N) v = B[r * N + c];
            Bs[ty][tx * TN + j] = v;
        }

        __syncthreads();

        // -----------------------------
        // 3) 在 shared 上做 BK 次 FMA
        // -----------------------------
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // A: 取本线程 8 行
            float a_frag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = __half2float(As[ty * TM + i][kk]);
            }

            // B: 取本线程 8 列
            float b_frag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = __half2float(Bs[kk][tx * TN + j]);
            }

            // outer product accumulate (8x8)
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }

        __syncthreads();
    }

    // -----------------------------
    // 4) 写回：C = alpha*acc + beta*C0
    // -----------------------------
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + i;
        if (r < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col0 + j;
                if (c < N) {
                    float c0 = __half2float(C[r * N + c]);
                    float out = alpha * acc[i][j] + beta * c0;
                    C[r * N + c] = __float2half(out);
                }
            }
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);
    gemm_tiled_smem<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

int main() {
    // 初始化 CUDA 相关参数
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;
    
    // 分配 host 内存
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(half);
    
    half *A_h = (half*)malloc(sizeA);
    half *B_h = (half*)malloc(sizeB);
    half *C_h = (half*)malloc(sizeC);
    
    // 分配 device 内存
    half *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeA);
    cudaMalloc(&B_d, sizeB);
    cudaMalloc(&C_d, sizeC);
    
    // 初始化 host 数据（简单示例）
    for (int i = 0; i < M * K; ++i) A_h[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) B_h[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; ++i) C_h[i] = __float2half(0.0f);
    
    // 复制到 device
    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, sizeC, cudaMemcpyHostToDevice);
    
    // 预热 + 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    solve(A_d, B_d, C_d, M, N, K, alpha, beta); // warmup
    
    cudaEventRecord(start);
    for (int it = 0; it < 100; ++it) {
        solve(A_d, B_d, C_d, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100.0f;
    
    printf("Average kernel time: %.4f ms\n", ms);
    
    // 拷贝结果回 host
    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}