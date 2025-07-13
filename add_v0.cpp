#include <iostream>
#include <math.h>

// https://developer.nvidia.com/zh-cn/blog/even-easier-introduction-cuda-2/

void add(int n, float *x, float *y){
    for(int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
}

int main(void){
    int N = 1<<20; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    //initialize x and y arrays on the host
    for (int i = 0; i < N ; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernel on 1M elements on the GPU
    add(N, x, y);

    float maxError = 0.0f;
    for(int i =0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }

    std::cout << "Max error:" << maxError << std::endl;

    //free memory
    delete[] x;
    delete[] y;

    return 0;

}