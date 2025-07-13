#include <math.h>

#include <iostream>

/*
https://developer.nvidia.com/zh-cn/blog/even-easier-introduction-cuda-2/
nvcc add_grid_v3.cu -o add_grid_v3
./add_grid_v3

分析：
./nsys_easy.sh ./add_grid_v3

Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max
(ns)   StdDev (ns)   Category                 Operation
 --------  ---------------  ---------  -----------  -----------  ---------
---------  -----------  -----------  ------------------------------------ 63.1
1,868,605          1  1,868,605.0  1,868,605.0  1,868,605  1,868,605 0.0
CUDA_KERNEL  add(int, float *, float *) 25.4          751,186         78 9,630.6
2,783.5      1,855     78,820     18,244.1  MEMORY_OPER  [CUDA memcpy Unified
Host-to-Device] 11.5          341,957         24     14,248.2      3,647.5 1,343
80,004     23,199.8  MEMORY_OPER  [CUDA memcpy Unified Device-to-Host]



对比结论：

性能从原来的114 ms	降到 1.87 ms ms	， 快约 37 倍。
*/

// kernel function to add the elements of two arrays
// threadIdx.x 包含其块内当前线程的索引，blockDim.x 包含块中的线程数

__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

// void add(int n, float *x, float *y) {
//   for (int i = 0; i < n; i++) {
//     y[i] = x[i] + y[i];
//   }
// }

int main(void) {
  int N = 1 << 20;  // 1M elements

  // allocate unified memory -- accessible from cpu or gpu
  float *x, *y, *sum;
  // 要在 GPU 上进行计算，我需要分配 GPU 可访问的内存。CUDA 中的 Unified Memory
  // 通过为系统中的所有 GPU 和 CPU 提供可访问的单一内存空间来简化这一过程。要在
  // Unified Memory 中分配数据，请调用 cudaMallocManaged()
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  /*
  run kernel on 1M elements on the GPU
  启动 add() 内核，以在 GPU 上调用它。CUDA 核函数启动使用三重角度括号语法  <<<
  >>> 指定。
  <<<1, 256>>>: 1个线程块，256线程块中的线程数


  <<<numBlocks, blockSize>>>并行线程块共同构成了所谓的 grid:
  线程块的计算方式：
  处理的元素个数：N
  每个线程块有256个线程，因此我只需计算线程块的数量即可获得至少 N 个线程
  */
  // // Prefetch the x and y arrays to the GPU
  // cudaMemPrefetchAsync(x, N * sizeof(float), 0, 0);
  // cudaMemPrefetchAsync(y, N * sizeof(float), 0, 0);
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // CPU 等到 kernel 完成后再访问结果 (因为 CUDA kernel 启动不会阻塞调用 CPU
  // 线程)
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }

  std::cout << "Max error:" << maxError << std::endl;

  // free memory
  //   要释放数据，只需将指针传递给 cudaFree() 即可。
  //   delete[] x;
  //   delete[] y;
  cudaFree(x);
  cudaFree(y);

  return 0;
}