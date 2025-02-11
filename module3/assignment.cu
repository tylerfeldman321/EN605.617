//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
using namespace std;

#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))

int cpu_a[ARRAY_SIZE];
int cpu_b[ARRAY_SIZE];
int cpu_result[ARRAY_SIZE];

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void add(int *result, int *a, int *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
	printf("Result[%d]: %d = %d + %d\n", i, result[i], a[i], b[i]);
  }
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;  // Also threads / block
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}


	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_a[i] = i;
	}

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 3);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_b[i] = distrib(gen);
	}

	printf("Array size: %d\n", ARRAY_SIZE);
	printf("Array size in bytes: %d\n", ARRAY_SIZE_IN_BYTES);

	int *gpu_a;
	int *gpu_b;
	int *gpu_result;

	checkCuda( cudaMalloc((void **)&gpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES) );

	cudaMemcpy( gpu_a, cpu_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_b, cpu_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	add<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
	checkCuda( cudaGetLastError() );
	checkCuda( cudaDeviceSynchronize() );

	checkCuda( cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost ) );

	printf("Result value at index 0: %d\n", cpu_result[0]);
	printf("Result value at index 1: %d\n", cpu_result[1]);
	printf("Result value at index 2: %d\n", cpu_result[2]);

	checkCuda( cudaFree(gpu_a) );
	checkCuda( cudaFree(gpu_b) );
	checkCuda( cudaFree(gpu_result) );
}
