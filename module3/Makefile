all: run_no_args assignment

profile: profile_diff_thread_block_combinations assignment

profile_diff_thread_block_combinations: assignment
	echo "Running profiling..."
	./assignment.exe 1048576 256
	./assignment.exe 524288 256
	./assignment.exe 262144 256
	./assignment.exe 1048576 256
	./assignment.exe 1048576 128
	./assignment.exe 1048576 32
	echo "Done profiling."

run_math_operations: assignment
	echo "Running math operation..."
	./assignment.exe 1024 256 add
	./assignment.exe 1024 256 subtract
	./assignment.exe 1024 256 multiply
	./assignment.exe 1024 256 mod
	echo "Done with math operations."

run_no_args: assignment
	./assignment.exe

assignment: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
