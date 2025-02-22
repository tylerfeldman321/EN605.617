#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std;

//From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char** argv)
{
  int n_exponent = 20;
  if (argc == 2) {
    n_exponent = atoi(argv[1]);
    cout << "[INFO] Setting n exponent to " << n_exponent << "\n";
  } else if (argc > 2) {
    cout << "[WARNING] Received more than 1 cli arguments, only 1 is supported.\n";
  }

  int N = 1<<n_exponent;
  cout << "[INFO] N=" << N << "\n";

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "[INFO] Time elapsed GPU = " << std::chrono::duration_cast<chrono::microseconds>(stop - start).count() << " microseconds\n";

  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(y[i]-4.0f));
    // printf("y[%d]=%f\n",i,y[i]);
  }
  printf("[INFO] Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
