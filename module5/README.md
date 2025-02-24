# Assignment 5: Memory

Please see comments in `assignment.cu` for a description of how different memory types are used. For this assignment, I created a complicated kernel that utilizes host, global, constant, shared, and register memory. 

## Running and Building

```bash
# Build the application and run it with a variety of thread and block combinations:
make

# Build the application and run it
make assignment
```

## Output of Profilling

```bash
Running profiling...

./assignment.exe 1048576 256
Array length: 1024, Array bytes: 4096, Blocks: 4096, Threads/block: 256, Total threads: 1048576
Results of operation: 
C[0]: 2, A[0]: 0, B[0], 1
C[1]: 10, A[1]: 1, B[1], 2
C[2]: 14, A[2]: 2, B[2], 1
C[3]: 20, A[3]: 3, B[3], 1
C[4]: 28, A[4]: 4, B[4], 2
Milliseconds elapsed: 0.249760

./assignment.exe 524288 256
Array length: 1024, Array bytes: 4096, Blocks: 2048, Threads/block: 256, Total threads: 524288
Results of operation: 
C[0]: 0, A[0]: 0, B[0], 0
C[1]: 6, A[1]: 1, B[1], 0
C[2]: 18, A[2]: 2, B[2], 3
C[3]: 22, A[3]: 3, B[3], 2
C[4]: 28, A[4]: 4, B[4], 2
Milliseconds elapsed: 0.312000

./assignment.exe 262144 256
Array length: 1024, Array bytes: 4096, Blocks: 1024, Threads/block: 256, Total threads: 262144
Results of operation: 
C[0]: 0, A[0]: 0, B[0], 0
C[1]: 6, A[1]: 1, B[1], 0
C[2]: 14, A[2]: 2, B[2], 1
C[3]: 24, A[3]: 3, B[3], 3
C[4]: 26, A[4]: 4, B[4], 1
Milliseconds elapsed: 0.225280

./assignment.exe 1048576 256
Array length: 1024, Array bytes: 4096, Blocks: 4096, Threads/block: 256, Total threads: 1048576
Results of operation: 
C[0]: 6, A[0]: 0, B[0], 3
C[1]: 10, A[1]: 1, B[1], 2
C[2]: 16, A[2]: 2, B[2], 2
C[3]: 22, A[3]: 3, B[3], 2
C[4]: 30, A[4]: 4, B[4], 3
Milliseconds elapsed: 0.230336

./assignment.exe 1048576 128
Array length: 1024, Array bytes: 4096, Blocks: 8192, Threads/block: 128, Total threads: 1048576
Results of operation: 
C[0]: 4, A[0]: 0, B[0], 2
C[1]: 6, A[1]: 1, B[1], 0
C[2]: 18, A[2]: 2, B[2], 3
C[3]: 20, A[3]: 3, B[3], 1
C[4]: 24, A[4]: 4, B[4], 0
Milliseconds elapsed: 0.242112

./assignment.exe 1048576 32
Array length: 1024, Array bytes: 4096, Blocks: 32768, Threads/block: 32, Total threads: 1048576
Results of operation: 
C[0]: 4, A[0]: 0, B[0], 2
C[1]: 6, A[1]: 1, B[1], 0
C[2]: 18, A[2]: 2, B[2], 3
C[3]: 18, A[3]: 3, B[3], 0
C[4]: 28, A[4]: 4, B[4], 2
Milliseconds elapsed: 0.278528

Done profiling.
```
