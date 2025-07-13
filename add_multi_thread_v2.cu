#include <math.h>

#include <iostream>

/*
https://developer.nvidia.com/zh-cn/blog/even-easier-introduction-cuda-2/
nvcc add_multi_thread_v2.cu -o add_multi_thread_v2
./add_multi_thread_v2

分析：
./nsys_easy.sh ./add_multi_thread_v2

Collecting data...
Max error:0
Generating '/tmp/nsys-report-3d2a.qdstrm'
[1/1] [========================100%] nsys_easy.nsys-rep
Generated:
        /home/s/codespace/cuda/nsys_easy.nsys-rep
Generating SQLite file nsys_easy.sqlite from nsys_easy.nsys-rep
Processing [nsys_easy.sqlite] with [/opt/nvidia/nsight-systems/2025.3.1/host-linux-x64/reports/cuda_gpu_sum.py]... 

 ** CUDA GPU Summary (Kernels/MemOps) (cuda_gpu_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Category                 Operation              
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -----------  ------------------------------------
     74.6        3,070,579          1  3,070,579.0  3,070,579.0  3,070,579  3,070,579          0.0  CUDA_KERNEL  add(int, float *, float *)          
     17.1          703,395         48     14,654.1      4,000.0      1,855     79,875     22,781.2  MEMORY_OPER  [CUDA memcpy Unified Host-to-Device]
      8.3          341,665         24     14,236.0      3,487.5      1,343     79,939     23,196.1  MEMORY_OPER  [CUDA memcpy Unified Device-to-Host]

对比结论：

执行从一个线程扩展到 256 个线程，性能从原来的114 ms	降到 3 ms	， 快约 37 倍。
*/



// kernel function to add the elements of two arrays
// threadIdx.x 包含其块内当前线程的索引，blockDim.x 包含块中的线程数
__global__ void add(int n, float *x, float *y) {
  int index = threadIdx.x;
  int stride = blockDim.x;
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

  // float *x = new float[N];
  // float *y = new float[N];

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

  // run kernel on 1M elements on the GPU
  //   启动 add() 内核，以在 GPU 上调用它。CUDA 核函数启动使用三重角度括号语法
  //   <<< >>> 指定。
  // <<<1, 256>>>: 指定1个线程块，256线程块中的线程数

  add<<<1, 256>>>(N, x, y);

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