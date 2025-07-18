/*

https://developer.nvidia.com/zh-cn/blog/how-optimize-data-transfers-cuda-cc/


nvcc profile_v2.cu -o profile_test 
nsys nvprof ./profile_test 

Collecting data...
Generating '/tmp/nsys-report-e9a0.qdstrm'
[1/7] [========================100%] report1.nsys-rep
[2/7] [========================100%] report1.sqlite
[3/7] Executing 'nvtx_sum' stats report
SKIPPED: /home/sunteng/codespace/cuda/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/7] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------
     99.2      102,467,576          1  102,467,576.0  102,467,576.0  102,467,576  102,467,576          0.0  cudaMalloc            
      0.8          821,901          2      410,950.5      410,950.5      290,611      531,290    170,185.8  cudaMemcpy            
      0.0            1,270          1        1,270.0        1,270.0        1,270        1,270          0.0  cuModuleGetLoadingMode

[5/7] Executing 'cuda_gpu_kern_sum' stats report
SKIPPED: /home/sunteng/codespace/cuda/report1.sqlite does not contain CUDA kernel data.
[6/7] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     50.5          327,665      1  327,665.0  327,665.0   327,665   327,665          0.0  [CUDA memcpy Device-to-Host]
     49.5          321,296      1  321,296.0  321,296.0   321,296   321,296          0.0  [CUDA memcpy Host-to-Device]

[7/7] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Device-to-Host]
      4.194      1     4.194     4.194     4.194     4.194        0.000  [CUDA memcpy Host-to-Device]

Generated:
        /home/sunteng/codespace/cuda/report1.nsys-rep
        /home/sunteng/codespace/cuda/report1.sqlite


        

*/
int main() {


  const unsigned int N = 1048576;
  const unsigned int bytes = N * sizeof(int);

  int *h_a = nullptr;
  int *d_a = nullptr;

   // 分配 pinned 内存
  cudaHostAlloc((void**)&h_a,bytes,cudaHostAllocDefault);

  // 分配 device 内存
  cudaMalloc((int**)&d_a, bytes);
  memset(h_a, 0, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  // 用 cudaFreeHost 释放
  cudaFreeHost(h_a);
  return 0;
}
