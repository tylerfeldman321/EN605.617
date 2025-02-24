//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace std;

#define ARRAY_SIZE (1 << 10)
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))

bool verbose = true;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// USING CONST MEMORY for broadcasting constant data to all threads
__constant__  static const int const_multiplication_factor = 2;

__global__
void complicated_math_kernel(int *c, int *a, int *b, int N)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // USING SHARED MEMORY for faster read/write memory shared between block
  // Copying data from c into shared data
  __shared__ int tmp_data[ARRAY_SIZE];
  for(int i = 0; i < ARRAY_SIZE; i++)
  {
	  tmp_data[i+tid] = c[i+tid];
  }
  __syncthreads();

  int stride = blockDim.x * gridDim.x;
  for(int i = tid; i < N; i += stride)
  {
	int tmp = c[i];  // USING REGISTER MEMORY for temp variables in kernel
    tmp_data[i] = a[i] + b[i];
	tmp_data[i] += tmp;
	tmp_data[i] *= const_multiplication_factor;
	c[i] = tmp_data[i];  // Copy data from shared memory back into global memory so host can access
  }
}

void initHostMemory(int* cpu_a, int* cpu_b, int* cpu_c) {
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_a[i] = i;
	}
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 3);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_b[i] = distrib(gen);
	}

	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_c[i] = i*2;
	}
}

void performMathOperations(int numBlocks, int blockSize, int totalThreads) {
	if (verbose) {
		printf("Array length: %d, Array bytes: %d, "
			"Blocks: %d, Threads/block: %d, Total threads: %d\n",
			(int)ARRAY_SIZE, (int)ARRAY_SIZE_IN_BYTES, 
			numBlocks, blockSize, totalThreads);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// USING GLOBAL MEMORY for IO to/from GPU
	int *gpu_a;
	int *gpu_b;
	int *gpu_c;
	checkCuda( cudaMalloc((void **)&gpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_c, ARRAY_SIZE_IN_BYTES) );

	// USING HOST MEMORY (of the pinned variety)
	int *cpu_a;
	int *cpu_b;
	int *cpu_c;
	checkCuda( cudaMallocHost((void **)&cpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMallocHost((void **)&cpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMallocHost((void **)&cpu_c, ARRAY_SIZE_IN_BYTES) );
	initHostMemory(cpu_a, cpu_b, cpu_c);

	cudaMemcpy( gpu_a, cpu_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_b, cpu_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_c, cpu_c, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	cudaEventRecord(start);
	complicated_math_kernel<<<numBlocks, blockSize>>>(gpu_c, gpu_a, gpu_b, ARRAY_SIZE);
	cudaEventRecord(stop);

	checkCuda( cudaDeviceSynchronize() );

	checkCuda( cudaGetLastError() );

	// Copy data back and synchronize
	checkCuda( cudaMemcpy( cpu_c, gpu_c, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost ) );
	checkCuda( cudaDeviceSynchronize() );
	if (verbose) {
		printf("Results of operation: \n");
		for (int i = 0; i < min(5, ARRAY_SIZE); i++) {
			printf("C[%d]: %d, A[%d]: %d, B[%d], %d\n", i, cpu_c[i], i, cpu_a[i], i, cpu_b[i]);
		}
	}

	// Print kernel runtime info
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Milliseconds elapsed: %f\n", milliseconds);

	// Free memory
	checkCuda( cudaFree(gpu_a) );
	checkCuda( cudaFree(gpu_b) );
	checkCuda( cudaFree(gpu_c) );
}


int main(int argc, char** argv)
{

	int totalThreads = (1 << 20);
	int blockSize = 256;  // Also threads / block

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	performMathOperations(numBlocks, blockSize, totalThreads);
}
