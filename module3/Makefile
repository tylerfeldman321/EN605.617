all: assignment profile

profile: assignment
	echo "Running profiling..."
	./assignment.exe 1048576 256
	./assignment.exe 524288 256
	./assignment.exe 262144 256
	./assignment.exe 1048576 256
	./assignment.exe 1048576 128
	./assignment.exe 1048576 32
	echo "Done profiling."

assignment: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
