//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>   
using namespace std;

#define ARRAY_SIZE 1024
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
	// printf("Result[%d]: %d = %d + %d\n", i, result[i], a[i], b[i]);
  }
}

__global__
void subtract(int *result, int *a, int *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] - b[i];
	// printf("Result[%d]: %d = %d + %d\n", i, result[i], a[i], b[i]);
  }
}

__global__
void multiply(int *result, int *a, int *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] * b[i];
	// printf("Result[%d]: %d = %d + %d\n", i, result[i], a[i], b[i]);
  }
}

__global__
void mod(int *result, int *a, int *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] % b[i];
	// printf("Result[%d]: %d = %d mod %d\n", i, result[i], a[i], b[i]);
  }
}


void initCpuArrays() {
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_a[i] = i;
	}

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 3);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cpu_b[i] = distrib(gen);
	}
}


void performMathOperations(int numBlocks, int blockSize, int totalThreads, std::string operation) {
	// performMathOperations()
	//  takes # blocks for kernel, block size (threads/block), total threads, and the math operation to do (add, subtract, multiply, or mod) and performs the operation

	initCpuArrays();
	printf("Op: %s, Array length: %d, Array bytes: %d, "
		"Blocks: %d, Threads/block: %d, Total threads: %d\n",
		operation.c_str(), (int)ARRAY_SIZE, (int)ARRAY_SIZE_IN_BYTES, 
		numBlocks, blockSize, totalThreads);

	int *gpu_a;
	int *gpu_b;
	int *gpu_result;

	checkCuda( cudaMalloc((void **)&gpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES) );

	cudaMemcpy( gpu_a, cpu_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_b, cpu_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> stop;
	if (operation == "add") {
		start = std::chrono::high_resolution_clock::now();
		add<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
		stop = std::chrono::high_resolution_clock::now();
	} else if (operation == "subtract") {
		start = std::chrono::high_resolution_clock::now();
		subtract<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
		stop = std::chrono::high_resolution_clock::now();
	} else if (operation == "multiply") {
		start = std::chrono::high_resolution_clock::now();
		multiply<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
		stop = std::chrono::high_resolution_clock::now();
	} else if (operation == "mod") {
		start = std::chrono::high_resolution_clock::now();
		mod<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
		stop = std::chrono::high_resolution_clock::now();
	} else {
		printf("Unexpected operation type: %s. Exiting...\n", operation.c_str());
		checkCuda( cudaFree(gpu_a) );
		checkCuda( cudaFree(gpu_b) );
		checkCuda( cudaFree(gpu_result) );
		exit(1);
	}

	checkCuda( cudaGetLastError() );
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost ) );

	std::cout << "Time elapsed GPU = " << std::chrono::duration_cast<chrono::nanoseconds>(stop - start).count() << " ns\n";
	printf("Results of operation: \n");
	for (int i = 0; i < min(5, ARRAY_SIZE); i++) {
		printf("Result[%d]: %d, A[%d]: %d, B[%d], %d\n", i, cpu_result[i], i, cpu_a[i], i, cpu_b[i]);
	}

	checkCuda( cudaFree(gpu_a) );
	checkCuda( cudaFree(gpu_b) );
	checkCuda( cudaFree(gpu_result) );
}


int main(int argc, char** argv)
{
	printf("----- New Run -----\n");

	int totalThreads = (1 << 20);
	int blockSize = 256;  // Also threads / block
	std::string operation("add");

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	if (argc >= 4) {
		operation = argv[3];
		std::cout << "Changed operation to " << operation << "\n";
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	performMathOperations(numBlocks, blockSize, totalThreads, operation);
}
