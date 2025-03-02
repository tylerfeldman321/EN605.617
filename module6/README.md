# Module 6 Assignment: 

For this assignment, I've 

## Building and Running
```bash
# Build and run the program
make run

# Build and profile
make profile
```

## Sample Results

```bash
ubuntu@ip-172-31-77-247:~/EN605.617/module6$ make
nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe -run
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
Array length: 33554432, Array bytes: 134217728, Blocks: 4096, Threads/block: 256, Total threads: 1048576
Milliseconds elapsed with streams: 71.542847
Results of operation with streams: 
C[0]: 1, A[0]: 0, B[0], 1
C[1]: 1, A[1]: 1, B[1], 0
C[2]: 5, A[2]: 2, B[2], 3
C[3]: 5, A[3]: 3, B[3], 2
C[4]: 6, A[4]: 4, B[4], 2
Milliseconds elapsed w/o streams: 86.268387
Results of operation w/o streams: 
C[0]: 2, A[0]: 0, B[0], 2
C[1]: 1, A[1]: 1, B[1], 0
C[2]: 5, A[2]: 2, B[2], 3
C[3]: 6, A[3]: 3, B[3], 3
```