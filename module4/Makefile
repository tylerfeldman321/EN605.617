all: host_memory global_memory

profile: profile_host_memory profile_global_memory

profile_host_memory: host_memory
	echo "Profiling host memory executable..."
	./profile.sh

profile_global_memory: global_memory
	echo "Profiling global memory executable..."
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 256/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 512/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 1024/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 2048/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 4096/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 8192/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 16384/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 32768/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run
	sed -i 's/^#define NUM_ELEMENTS .*/#define NUM_ELEMENTS 65536/' global_memory.cu
	nvcc -o global_memory global_memory.cu -run	

host_memory: host_memory.cu
	nvcc -o host_memory host_memory.cu

global_memory: global_memory.cu
	nvcc -o global_memory global_memory.cu

clean:
	rm -f host_memory
	rm -f global_memory
