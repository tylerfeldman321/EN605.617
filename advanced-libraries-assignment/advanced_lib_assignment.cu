#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/count.h>

#include <stdio.h>
#include <curand.h>
#include <cuda_runtime.h>


// From https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuRAND/utils/curand_utils.h and other official curand examples from nvidia
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


struct withinQuarterCircle
{
    __host__ __device__
    bool operator()(const thrust::tuple<float, float>& xy) const
    {
        float x = thrust::get<0>(xy);
        float y = thrust::get<1>(xy);
        return (x * x + y * y) <= 1.0f;
    }
};


int main(int argc, char** argv) {
    int pointsToSample = (1 << 20);
    if (argc >= 2) {
        char* endptr;
        long input = strtol(argv[1], &endptr, 10);

        // Check for conversion errors
        if (*endptr != '\0') {
            std::cerr << "Error: Invalid number format.\n";
            return 1;
        }

        // Check for overflow or underflow
        if (input > INT_MAX || input < INT_MIN) {
            std::cerr << "Error: Input out of range for int.\n";
            return 1;
        }

        pointsToSample = static_cast<int>(input);
    }
    printf("Using %d points to sample\n", pointsToSample);

    // Timing setup
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

    // cuRAND initialization
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
            1234ULL));

    // Device memory allocation
    float *devDataX, *devDataY;
    CUDA_CALL(cudaMalloc((void **)&devDataX, pointsToSample*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&devDataY, pointsToSample*sizeof(float)));

    // Uniform data generation
    CURAND_CALL(curandGenerateUniform(gen, devDataX, pointsToSample));
    CURAND_CALL(curandGenerateUniform(gen, devDataY, pointsToSample));

    // Copy data into thrust device vectors
    thrust::device_vector<float> d_x(devDataX, devDataX + pointsToSample);
    thrust::device_vector<float> d_y(devDataY, devDataY + pointsToSample);

    // Create iterators through x and y points, count if they are in quarter circle, compute ratio of points, multiply by 4 to compute pi
    // reference: https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end()));
    int insideCircleCount = thrust::count_if(begin, end, withinQuarterCircle());
    float estimateOfPi = 4.0f * static_cast<float>(insideCircleCount) / static_cast<float>(pointsToSample);

	cudaEventRecord(stop);

    printf("Estimate of Pi: %f\n", estimateOfPi);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Milliseconds elapsed: %f\n", milliseconds);

    // Cleanup of cuRAND and device memory. Thrust vectors should clean themselves up
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devDataX));
    CUDA_CALL(cudaFree(devDataY));
}