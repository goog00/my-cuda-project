/*
https://developer.nvidia.com/zh-cn/blog/how-optimize-data-transfers-cuda-cc/

nvcc bandwidthtest.cu -o bandwidthtest
nsys nvprof ./bandwidthtest
WARNING: bandwidthtest and any of its children processes will be profiled.

Collecting data...

Device: NVIDIA GeForce RTX 3050
Transfer size (MB): 16

Pageable transfers
  Host to Device bandwidth (GB/s): 10.672529
  Device to Host bandwidth (GB/s): 11.632749

Pinned transfers
  Host to Device bandwidth (GB/s): 12.990609
  Device to Host bandwidth (GB/s): 13.080711
nGenerating '/tmp/nsys-report-4e87.qdstrm'
[1/7] [========================100%] report1.nsys-rep
[2/7] [========================100%] report1.sqlite
[3/7] Executing 'nvtx_sum' stats report
SKIPPED: /home/sunteng/codespace/cuda/data-transfer/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/7] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)                 Name               
 --------  ---------------  ---------  ------------  ------------  ----------  -----------  ------------  ---------------------------------
     83.9      125,703,762          2  62,851,881.0  62,851,881.0  10,124,820  115,578,942  74,567,324.8  cudaMallocHost                   
      5.6        8,343,304          4   2,085,826.0       2,355.0         380    8,338,214   4,168,259.0  cudaEventCreate                  
      5.4        8,159,272          2   4,079,636.0   4,079,636.0   3,806,644    4,352,628     386,069.0  cudaFreeHost                     
      3.7        5,486,288          4   1,371,572.0   1,360,547.0   1,279,301    1,485,893     106,344.8  cudaMemcpy                       
      1.1        1,720,705          1   1,720,705.0   1,720,705.0   1,720,705    1,720,705           0.0  cudaGetDeviceProperties_v2_v12000
      0.1          179,762          1     179,762.0     179,762.0     179,762      179,762           0.0  cudaFree                         
      0.1          112,411          1     112,411.0     112,411.0     112,411      112,411           0.0  cudaMalloc                       
      0.1           80,360          4      20,090.0       1,970.0       1,830       74,590      36,333.5  cudaEventSynchronize             
      0.0           59,681          8       7,460.1       4,130.0       2,970       24,200       7,224.0  cudaEventRecord                  
      0.0            8,620          4       2,155.0       2,095.0         330        4,100       1,859.4  cudaEventDestroy                 
      0.0              900          1         900.0         900.0         900          900           0.0  cuModuleGetLoadingMode           

[5/7] Executing 'cuda_gpu_kern_sum' stats report
SKIPPED: /home/sunteng/codespace/cuda/data-transfer/report1.sqlite does not contain CUDA kernel data.
[6/7] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)           Operation          
 --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ----------------------------
     50.9        2,710,235      2  1,355,117.5  1,355,117.5  1,259,784  1,450,451    134,821.9  [CUDA memcpy Host-to-Device]
     49.1        2,613,173      2  1,306,586.5  1,306,586.5  1,271,241  1,341,932     49,986.1  [CUDA memcpy Device-to-Host]

[7/7] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
     33.554      2    16.777    16.777    16.777    16.777        0.000  [CUDA memcpy Device-to-Host]
     33.554      2    16.777    16.777    16.777    16.777        0.000  [CUDA memcpy Host-to-Device]

Generated:
        /home/sunteng/codespace/cuda/data-transfer/report1.nsys-rep
        /home/sunteng/codespace/cuda/data-transfer/report1.sqlite
*/

#include <assert.h>
#include <stdio.h>

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error: %s\n", cudaGetErrorString(result));

    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n,
                   char *desc) {
 printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

int main() {
  unsigned int nElements = 4 * 1024 * 1024;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  // device array
  float *d_a;

  // allocate and initialize
  h_aPageable = (float *)malloc(bytes);
  h_bPageable = (float *)malloc(bytes);

  checkCuda(cudaMallocHost((void **)&h_aPinned, bytes));
  checkCuda(cudaMallocHost((void **)&h_bPinned, bytes));
  checkCuda(cudaMalloc((void **)&d_a, bytes));

  for (int i = 0; i < nElements; ++i) {
    h_aPageable[i] = i;
  }

  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  //
  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

  printf("n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;

}