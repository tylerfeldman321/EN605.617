all: assignment profile

profile: assignment
	echo "Running profiling..."
	./assignment.exe
	echo "Done profiling."

assignment: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
