//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace std;

#define ARRAY_SIZE (1 << 25)
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))
#define NUM_STREAMS 8

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void work_kernel(int *c, int *a, int *b, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = tid; i < N; i+=stride) {
	c[i] = a[i] + b[i];
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
		cpu_c[i] = 0;
	}
}


void executeKernelWithNoStreams(int numBlocks, int blockSize, int* gpu_a, int* gpu_b, int* gpu_c, int* cpu_a, int* cpu_b, int* cpu_c, bool printResults) {
	printf("---- Executing kernel w/o streams ----\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	checkCuda( cudaMemcpy( gpu_a, cpu_a, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice ) );
	checkCuda( cudaMemcpy( gpu_b, cpu_b, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice ) );
	checkCuda( cudaMemcpy( gpu_c, cpu_c, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice ) );
	work_kernel<<<numBlocks, blockSize>>>(gpu_c, gpu_a, gpu_b, ARRAY_SIZE);
	checkCuda( cudaMemcpy( cpu_c, gpu_c, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost ) );
	cudaEventRecord(stop);

	checkCuda( cudaGetLastError() );
	checkCuda( cudaDeviceSynchronize() );

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Milliseconds elapsed w/o streams: %f\n", milliseconds);

	if (printResults) {
		printf("Results of operation with streams: \n");
		for (int i = 0; i < min(10, ARRAY_SIZE); i++) {
			printf("C[%d]: %d, A[%d]: %d, B[%d], %d\n", i, cpu_c[i], i, cpu_a[i], i, cpu_b[i]);
		}
		printf("...\n");
	}
}


void executeKernelWithStreams(int numBlocks, int blockSize, int* gpu_a, int* gpu_b, int* gpu_c, int* cpu_a, int* cpu_b, int* cpu_c, bool printResults) {
	printf("---- Executing kernel with streams ----\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int dataPerStream = ARRAY_SIZE / NUM_STREAMS;
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(streams+i);
	}

	for (int streamId = 0; streamId < NUM_STREAMS; streamId++) {
		int streamDataOffset = streamId * dataPerStream;
		checkCuda( cudaMemcpyAsync( gpu_a+streamDataOffset, cpu_a+streamDataOffset, dataPerStream*sizeof(int), cudaMemcpyHostToDevice, streams[streamId]) );
		checkCuda( cudaMemcpyAsync( gpu_b+streamDataOffset, cpu_b+streamDataOffset, dataPerStream*sizeof(int), cudaMemcpyHostToDevice, streams[streamId] ) );
		checkCuda( cudaMemcpyAsync( gpu_c+streamDataOffset, cpu_c+streamDataOffset, dataPerStream*sizeof(int), cudaMemcpyHostToDevice, streams[streamId] ) );
		work_kernel<<<numBlocks, blockSize, 0, streams[streamId]>>>(gpu_c+streamDataOffset, gpu_a+streamDataOffset, gpu_b+streamDataOffset, dataPerStream);
		checkCuda( cudaMemcpyAsync( cpu_c+streamDataOffset, gpu_c+streamDataOffset, dataPerStream*sizeof(int), cudaMemcpyDeviceToHost, streams[streamId]) );
		cudaStreamDestroy(streams[streamId]);
	}

	cudaEventRecord(stop);

	checkCuda( cudaGetLastError() );
	checkCuda( cudaDeviceSynchronize() );

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Milliseconds elapsed with streams: %f\n", milliseconds);

	if (printResults) {
		printf("Results of operation with streams: \n");
		for (int i = 0; i < min(10, ARRAY_SIZE); i++) {
			printf("C[%d]: %d, A[%d]: %d, B[%d], %d\n", i, cpu_c[i], i, cpu_a[i], i, cpu_b[i]);
		}
		printf("...\n");
	}
}

void performWork(int numBlocks, int blockSize, int totalThreads, bool printResults) {
	printf("Array length: %d, Array bytes: %d, "
		"Blocks: %d, Threads/block: %d, Total threads: %d\n",
		(int)ARRAY_SIZE, (int)ARRAY_SIZE_IN_BYTES, 
		numBlocks, blockSize, totalThreads);

	int *gpu_a;
	int *gpu_b;
	int *gpu_c;
	checkCuda( cudaMalloc((void **)&gpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMalloc((void **)&gpu_c, ARRAY_SIZE_IN_BYTES) );

	int *cpu_a;
	int *cpu_b;
	int *cpu_c;
	checkCuda( cudaMallocHost((void **)&cpu_a, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMallocHost((void **)&cpu_b, ARRAY_SIZE_IN_BYTES) );
	checkCuda( cudaMallocHost((void **)&cpu_c, ARRAY_SIZE_IN_BYTES) );
	
	// Method 1: Using streams
	initHostMemory(cpu_a, cpu_b, cpu_c);
	executeKernelWithStreams(numBlocks, blockSize, gpu_a, gpu_b, gpu_c, cpu_a, cpu_b, cpu_c, printResults);

	// Method 2: Not using streams
	initHostMemory(cpu_a, cpu_b, cpu_c);
	executeKernelWithNoStreams(numBlocks, blockSize, gpu_a, gpu_b, gpu_c, cpu_a, cpu_b, cpu_c, printResults);

	checkCuda( cudaFree(gpu_a) );
	checkCuda( cudaFree(gpu_b) );
	checkCuda( cudaFree(gpu_c) );
}


int main(int argc, char** argv)
{
	int totalThreads = (1 << 20);
	int blockSize = 256;  // Also threads / block
	bool printResults = false;
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	if (argc >= 4) {
		printResults = atoi(argv[3]);
	}

	int numBlocks = totalThreads/blockSize;

	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	performWork(numBlocks, blockSize, totalThreads, printResults);
}
