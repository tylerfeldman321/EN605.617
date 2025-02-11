//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace std;

#define ARRAY_SIZE 20480
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))

int cpu_a[ARRAY_SIZE];
int cpu_b[ARRAY_SIZE];
int cpu_result[ARRAY_SIZE];
bool verbose = false;

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

__global__
void branchingKernel(int *result, int *a, int *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < N; i += stride)
  {
	if (i % 2 == 0) {
    	result[i] = a[i] + b[i];
	} else {
		result[i] = a[i] * b[i];
	}
  }
}


void initCpuArrays() {
	// Initializes cpu_a data to 0...N and cpu_b data to random numbers from 0-3 inclusive
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
	if (verbose) {
		printf("----- Math Operations -----\n");
		printf("Op: %s, Array length: %d, Array bytes: %d, "
			"Blocks: %d, Threads/block: %d, Total threads: %d\n",
			operation.c_str(), (int)ARRAY_SIZE, (int)ARRAY_SIZE_IN_BYTES, 
			numBlocks, blockSize, totalThreads);
	}
	initCpuArrays();

	int *gpu_a;
	int *gpu_b;
	int *gpu_result;
	checkCuda( cudaMalloc((void **)&gpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES) );

	cudaMemcpy( gpu_a, cpu_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_b, cpu_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	// Perform and time the operation, synchronizing before stopping the timer
	auto start = std::chrono::high_resolution_clock::now();
	if (operation == "add") {
		add<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
	} else if (operation == "subtract") {
		subtract<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
	} else if (operation == "multiply") {
		multiply<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
	} else if (operation == "mod") {
		mod<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
	} else {
		printf("Unexpected operation type: %s. Exiting...\n", operation.c_str());
		checkCuda( cudaFree(gpu_a) );
		checkCuda( cudaFree(gpu_b) );
		checkCuda( cudaFree(gpu_result) );
		exit(1);
	}
	checkCuda( cudaDeviceSynchronize() );
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Time elapsed GPU = " << std::chrono::duration_cast<chrono::nanoseconds>(stop - start).count() << " ns\n";

	checkCuda( cudaGetLastError() );

	// Copy data back and synchronize
	checkCuda( cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost ) );
	checkCuda( cudaDeviceSynchronize() );
	if (verbose) {
		printf("Results of operation: \n");
		for (int i = 0; i < min(5, ARRAY_SIZE); i++) {
			printf("Result[%d]: %d, A[%d]: %d, B[%d], %d\n", i, cpu_result[i], i, cpu_a[i], i, cpu_b[i]);
		}
	}

	checkCuda( cudaFree(gpu_a) );
	checkCuda( cudaFree(gpu_b) );
	checkCuda( cudaFree(gpu_result) );
}


void demonstrateConditionalBranching(int numBlocks, int blockSize, int totalThreads) {
	if (verbose) {
		printf("----- Conditional Branching -----\n");
		printf("Conditional branching kernel, Array length: %d, Array bytes: %d, "
			"Blocks: %d, Threads/block: %d, Total threads: %d\n",
			(int)ARRAY_SIZE, (int)ARRAY_SIZE_IN_BYTES, 
			numBlocks, blockSize, totalThreads);
	}

	initCpuArrays();
	int *gpu_a;
	int *gpu_b;
	int *gpu_result;

	checkCuda( cudaMalloc((void **)&gpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES) );

	cudaMemcpy( gpu_a, cpu_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_b, cpu_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	// Perform and time the operation, synchronizing before stopping the timer
	auto start = std::chrono::high_resolution_clock::now();
	branchingKernel<<<numBlocks, blockSize>>>(gpu_result, gpu_a, gpu_b, ARRAY_SIZE);
	checkCuda( cudaDeviceSynchronize() );
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Time elapsed GPU = " << std::chrono::duration_cast<chrono::nanoseconds>(stop - start).count() << " ns\n";

	checkCuda( cudaGetLastError() );

	// Copy data back and synchronize
	checkCuda( cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost ) );
	checkCuda( cudaDeviceSynchronize() );
	checkCuda( cudaFree(gpu_a) );
	checkCuda( cudaFree(gpu_b) );
	checkCuda( cudaFree(gpu_result) );
}


int main(int argc, char** argv)
{

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
		if (verbose)
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

	// printf("Performing warm up run...\n");
	performMathOperations(numBlocks, blockSize, totalThreads, operation);
	
	// printf("Performing real run...\n");
	performMathOperations(numBlocks, blockSize, totalThreads, operation);

	demonstrateConditionalBranching(numBlocks, blockSize, totalThreads);
}
