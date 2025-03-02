all: assignment.exe run

profile: profile_diff_thread_block_combinations

profile_diff_thread_block_combinations: assignment.exe
	echo "Running profiling..."
	./assignment.exe 1048576 256
	./assignment.exe 524288 256
	./assignment.exe 262144 256
	./assignment.exe 1048576 128
	./assignment.exe 1048576 32
	echo "Done profiling."

run: assignment.exe
	./assignment.exe 1048576 256 1

assignment.exe: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe

clean:
	rm assignment.exe
